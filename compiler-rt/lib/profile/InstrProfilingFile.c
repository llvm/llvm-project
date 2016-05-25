/*===- InstrProfilingFile.c - Write instrumentation to a file -------------===*\
|*
|*                     The LLVM Compiler Infrastructure
|*
|* This file is distributed under the University of Illinois Open Source
|* License. See LICENSE.TXT for details.
|*
\*===----------------------------------------------------------------------===*/

#include "InstrProfiling.h"
#include "InstrProfilingInternal.h"
#include "InstrProfilingUtil.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _MSC_VER
/* For _alloca. */
#include <malloc.h>
#endif

#define MAX_PID_SIZE 16
/* Data structure holding the result of parsed filename pattern.  */
typedef struct lprofFilename {
  /* File name string possibly with %p or %h specifiers. */
  const char *FilenamePat;
  char PidChars[MAX_PID_SIZE];
  char Hostname[COMPILER_RT_MAX_HOSTLEN];
  unsigned NumPids;
  unsigned NumHosts;
} lprofFilename;

lprofFilename lprofCurFilename = {0, {0}, {0}, 0, 0};

int getpid(void);
static int getCurFilenameLength();
static const char *getCurFilename(char *FilenameBuf);

/* Return 1 if there is an error, otherwise return  0.  */
static uint32_t fileWriter(ProfDataIOVec *IOVecs, uint32_t NumIOVecs,
                           void **WriterCtx) {
  uint32_t I;
  FILE *File = (FILE *)*WriterCtx;
  for (I = 0; I < NumIOVecs; I++) {
    if (fwrite(IOVecs[I].Data, IOVecs[I].ElmSize, IOVecs[I].NumElm, File) !=
        IOVecs[I].NumElm)
      return 1;
  }
  return 0;
}

COMPILER_RT_VISIBILITY ProfBufferIO *
lprofCreateBufferIOInternal(void *File, uint32_t BufferSz) {
  FreeHook = &free;
  DynamicBufferIOBuffer = (uint8_t *)calloc(BufferSz, 1);
  VPBufferSize = BufferSz;
  return lprofCreateBufferIO(fileWriter, File);
}

static void setupIOBuffer() {
  const char *BufferSzStr = 0;
  BufferSzStr = getenv("LLVM_VP_BUFFER_SIZE");
  if (BufferSzStr && BufferSzStr[0]) {
    VPBufferSize = atoi(BufferSzStr);
    DynamicBufferIOBuffer = (uint8_t *)calloc(VPBufferSize, 1);
  }
}

/* Write profile data to file \c OutputName.  */
static int writeFile(const char *OutputName) {
  int RetVal;
  FILE *OutputFile;

  /* Append to the file to support profiling multiple shared objects. */
  OutputFile = fopen(OutputName, "ab");
  if (!OutputFile)
    return -1;

  FreeHook = &free;
  setupIOBuffer();
  RetVal = lprofWriteData(fileWriter, OutputFile, lprofGetVPDataReader());

  fclose(OutputFile);
  return RetVal;
}

static void truncateCurrentFile(void) {
  const char *Filename;
  char *FilenameBuf;
  FILE *File;
  int Length;

  Length = getCurFilenameLength();
  FilenameBuf = (char *)COMPILER_RT_ALLOCA(Length + 1);
  Filename = getCurFilename(FilenameBuf);
  if (!Filename)
    return;

  /* Create the directory holding the file, if needed. */
  if (strchr(Filename, '/') || strchr(Filename, '\\')) {
    char *Copy = (char *)COMPILER_RT_ALLOCA(Length + 1);
    strncpy(Copy, Filename, Length + 1);
    __llvm_profile_recursive_mkdir(Copy);
  }

  /* Truncate the file.  Later we'll reopen and append. */
  File = fopen(Filename, "w");
  if (!File)
    return;
  fclose(File);
}

static void resetFilenameToDefault(void) {
  memset(&lprofCurFilename, 0, sizeof(lprofCurFilename));
  lprofCurFilename.FilenamePat = "default.profraw";
}

/* Parses the pattern string \p FilenamePat and store the result to
 * lprofcurFilename structure. */

static int parseFilenamePattern(const char *FilenamePat) {
  int NumPids = 0, NumHosts = 0, I;
  char *PidChars = &lprofCurFilename.PidChars[0];
  char *Hostname = &lprofCurFilename.Hostname[0];

  lprofCurFilename.FilenamePat = FilenamePat;
  /* Check the filename for "%p", which indicates a pid-substitution. */
  for (I = 0; FilenamePat[I]; ++I)
    if (FilenamePat[I] == '%') {
      if (FilenamePat[++I] == 'p') {
        if (!NumPids++) {
          if (snprintf(PidChars, MAX_PID_SIZE, "%d", getpid()) <= 0) {
            PROF_WARN(
                "Unable to parse filename pattern %s. Using the default name.",
                FilenamePat);
            return -1;
          }
        }
      } else if (FilenamePat[I] == 'h') {
        if (!NumHosts++)
          if (COMPILER_RT_GETHOSTNAME(Hostname, COMPILER_RT_MAX_HOSTLEN)) {
            PROF_WARN(
                "Unable to parse filename pattern %s. Using the default name.",
                FilenamePat);
            return -1;
          }
      }
    }

  lprofCurFilename.NumPids = NumPids;
  lprofCurFilename.NumHosts = NumHosts;
  return 0;
}

static void parseAndSetFilename(const char *FilenamePat) {
  int NewFile;
  const char *OldFilenamePat = lprofCurFilename.FilenamePat;

  if (!FilenamePat || parseFilenamePattern(FilenamePat))
    resetFilenameToDefault();

  NewFile =
      !OldFilenamePat || (strcmp(OldFilenamePat, lprofCurFilename.FilenamePat));

  if (NewFile)
    truncateCurrentFile();
}

/* Return buffer length that is required to store the current profile
 * filename with PID and hostname substitutions. */
static int getCurFilenameLength() {
  if (!lprofCurFilename.FilenamePat || !lprofCurFilename.FilenamePat[0])
    return 0;

  if (!(lprofCurFilename.NumPids || lprofCurFilename.NumHosts))
    return strlen(lprofCurFilename.FilenamePat);

  return strlen(lprofCurFilename.FilenamePat) +
         lprofCurFilename.NumPids * (strlen(lprofCurFilename.PidChars) - 2) +
         lprofCurFilename.NumHosts * (strlen(lprofCurFilename.Hostname) - 2);
}

/* Return the pointer to the current profile file name (after substituting
 * PIDs and Hostnames in filename pattern. \p FilenameBuf is the buffer
 * to store the resulting filename. If no substitution is needed, the
 * current filename pattern string is directly returned. */
static const char *getCurFilename(char *FilenameBuf) {
  int I, J, PidLength, HostNameLength;
  const char *FilenamePat = lprofCurFilename.FilenamePat;

  if (!lprofCurFilename.FilenamePat || !lprofCurFilename.FilenamePat[0])
    return 0;

  if (!(lprofCurFilename.NumPids || lprofCurFilename.NumHosts))
    return lprofCurFilename.FilenamePat;

  PidLength = strlen(lprofCurFilename.PidChars);
  HostNameLength = strlen(lprofCurFilename.Hostname);
  /* Construct the new filename. */
  for (I = 0, J = 0; FilenamePat[I]; ++I)
    if (FilenamePat[I] == '%') {
      if (FilenamePat[++I] == 'p') {
        memcpy(FilenameBuf + J, lprofCurFilename.PidChars, PidLength);
        J += PidLength;
      } else if (FilenamePat[I] == 'h') {
        memcpy(FilenameBuf + J, lprofCurFilename.Hostname, HostNameLength);
        J += HostNameLength;
      }
      /* Drop any unknown substitutions. */
    } else
      FilenameBuf[J++] = FilenamePat[I];
  FilenameBuf[J] = 0;

  return FilenameBuf;
}

/* Returns the pointer to the environment variable
 * string. Returns null if the env var is not set. */
static const char *getFilenamePatFromEnv(void) {
  const char *Filename = getenv("LLVM_PROFILE_FILE");
  if (!Filename || !Filename[0])
    return 0;
  return Filename;
}

/* This method is invoked by the runtime initialization hook
 * InstrProfilingRuntime.o if it is linked in. Both user specified
 * profile path via -fprofile-instr-generate= and LLVM_PROFILE_FILE
 * environment variable can override this default value. */
COMPILER_RT_VISIBILITY
void __llvm_profile_initialize_file(void) {
  const char *FilenamePat;
  /* Check if the filename has been initialized. */
  if (lprofCurFilename.FilenamePat)
    return;

  /* Detect the filename and truncate. */
  FilenamePat = getFilenamePatFromEnv();
  parseAndSetFilename(FilenamePat);
}

/* This API is directly called by the user application code. It has the
 * highest precedence compared with LLVM_PROFILE_FILE environment variable
 * and command line option -fprofile-instr-generate=<profile_name>.
 */
COMPILER_RT_VISIBILITY
void __llvm_profile_set_filename(const char *FilenamePat) {
  parseAndSetFilename(FilenamePat);
}

/*
 * This API is invoked by the global initializers emitted by Clang/LLVM when
 * -fprofile-instr-generate=<..> is specified (vs -fprofile-instr-generate
 *  without an argument). This option has lower precedence than the
 *  LLVM_PROFILE_FILE environment variable.
 */
COMPILER_RT_VISIBILITY
void __llvm_profile_override_default_filename(const char *FilenamePat) {
  /* If the env var is set, skip setting filename from argument. */
  const char *Env_Filename = getFilenamePatFromEnv();
  if (Env_Filename)
    return;

  parseAndSetFilename(FilenamePat);
}

/* The public API for writing profile data into the file with name
 * set by previous calls to __llvm_profile_set_filename or
 * __llvm_profile_override_default_filename or
 * __llvm_profile_initialize_file. */
COMPILER_RT_VISIBILITY
int __llvm_profile_write_file(void) {
  int rc, Length;
  const char *Filename;
  char *FilenameBuf;

  Length = getCurFilenameLength();
  FilenameBuf = (char *)COMPILER_RT_ALLOCA(Length + 1);
  Filename = getCurFilename(FilenameBuf);

  /* Check the filename. */
  if (!Filename) {
    PROF_ERR("Failed to write file : %s\n", "Filename not set");
    return -1;
  }

  /* Check if there is llvm/runtime version mismatch.  */
  if (GET_VERSION(__llvm_profile_get_version()) != INSTR_PROF_RAW_VERSION) {
    PROF_ERR("Runtime and instrumentation version mismatch : "
             "expected %d, but get %d\n",
             INSTR_PROF_RAW_VERSION,
             (int)GET_VERSION(__llvm_profile_get_version()));
    return -1;
  }

  /* Write profile data to the file. */
  rc = writeFile(Filename);
  if (rc)
    PROF_ERR("Failed to write file \"%s\": %s\n", Filename, strerror(errno));
  return rc;
}

static void writeFileWithoutReturn(void) { __llvm_profile_write_file(); }

COMPILER_RT_VISIBILITY
int __llvm_profile_register_write_file_atexit(void) {
  static int HasBeenRegistered = 0;

  if (HasBeenRegistered)
    return 0;

  lprofSetupValueProfiler();

  HasBeenRegistered = 1;
  return atexit(writeFileWithoutReturn);
}
