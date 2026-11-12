// RUN: %clang_profgen -o %t.exe %s
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: cd %t.dir
// RUN: %run %t.exe

#include <stdio.h>
#include <windows.h>

extern FILE *lprofOpenFileEx(const char *);

int main(void) {
  const char *Filename = "profile-\xe6\x97\xa5.dump";
  FILE *File = lprofOpenFileEx(Filename);
  if (!File)
    return 1;

  fputs("profile data", File);
  fclose(File);

  return GetFileAttributesW(L"profile-\u65e5.dump") == INVALID_FILE_ATTRIBUTES;
}
