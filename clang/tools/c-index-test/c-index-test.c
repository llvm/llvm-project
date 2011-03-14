/* c-index-test.c */

#include "clang-c/Index.h"
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

/******************************************************************************/
/* Utility functions.                                                         */
/******************************************************************************/

#ifdef _MSC_VER
char *basename(const char* path)
{
    char* base1 = (char*)strrchr(path, '/');
    char* base2 = (char*)strrchr(path, '\\');
    if (base1 && base2)
        return((base1 > base2) ? base1 + 1 : base2 + 1);
    else if (base1)
        return(base1 + 1);
    else if (base2)
        return(base2 + 1);

    return((char*)path);
}
#else
extern char *basename(const char *);
#endif

/** \brief Return the default parsing options. */
static unsigned getDefaultParsingOptions() {
  unsigned options = CXTranslationUnit_DetailedPreprocessingRecord;

  if (getenv("CINDEXTEST_EDITING"))
    options |= clang_defaultEditingTranslationUnitOptions();
  if (getenv("CINDEXTEST_COMPLETION_CACHING"))
    options |= CXTranslationUnit_CacheCompletionResults;
  
  return options;
}

static void PrintExtent(FILE *out, unsigned begin_line, unsigned begin_column,
                        unsigned end_line, unsigned end_column) {
  fprintf(out, "[%d:%d - %d:%d]", begin_line, begin_column,
          end_line, end_column);
}

static unsigned CreateTranslationUnit(CXIndex Idx, const char *file,
                                      CXTranslationUnit *TU) {

  *TU = clang_createTranslationUnit(Idx, file);
  if (!*TU) {
    fprintf(stderr, "Unable to load translation unit from '%s'!\n", file);
    return 0;
  }
  return 1;
}

void free_remapped_files(struct CXUnsavedFile *unsaved_files,
                         int num_unsaved_files) {
  int i;
  for (i = 0; i != num_unsaved_files; ++i) {
    free((char *)unsaved_files[i].Filename);
    free((char *)unsaved_files[i].Contents);
  }
  free(unsaved_files);
}

int parse_remapped_files(int argc, const char **argv, int start_arg,
                         struct CXUnsavedFile **unsaved_files,
                         int *num_unsaved_files) {
  int i;
  int arg;
  int prefix_len = strlen("-remap-file=");
  *unsaved_files = 0;
  *num_unsaved_files = 0;

  /* Count the number of remapped files. */
  for (arg = start_arg; arg < argc; ++arg) {
    if (strncmp(argv[arg], "-remap-file=", prefix_len))
      break;

    ++*num_unsaved_files;
  }

  if (*num_unsaved_files == 0)
    return 0;

  *unsaved_files
    = (struct CXUnsavedFile *)malloc(sizeof(struct CXUnsavedFile) *
                                     *num_unsaved_files);
  for (arg = start_arg, i = 0; i != *num_unsaved_files; ++i, ++arg) {
    struct CXUnsavedFile *unsaved = *unsaved_files + i;
    const char *arg_string = argv[arg] + prefix_len;
    int filename_len;
    char *filename;
    char *contents;
    FILE *to_file;
    const char *semi = strchr(arg_string, ';');
    if (!semi) {
      fprintf(stderr,
              "error: -remap-file=from;to argument is missing semicolon\n");
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }

    /* Open the file that we're remapping to. */
    to_file = fopen(semi + 1, "rb");
    if (!to_file) {
      fprintf(stderr, "error: cannot open file %s that we are remapping to\n",
              semi + 1);
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }

    /* Determine the length of the file we're remapping to. */
    fseek(to_file, 0, SEEK_END);
    unsaved->Length = ftell(to_file);
    fseek(to_file, 0, SEEK_SET);

    /* Read the contents of the file we're remapping to. */
    contents = (char *)malloc(unsaved->Length + 1);
    if (fread(contents, 1, unsaved->Length, to_file) != unsaved->Length) {
      fprintf(stderr, "error: unexpected %s reading 'to' file %s\n",
              (feof(to_file) ? "EOF" : "error"), semi + 1);
      fclose(to_file);
      free_remapped_files(*unsaved_files, i);
      *unsaved_files = 0;
      *num_unsaved_files = 0;
      return -1;
    }
    contents[unsaved->Length] = 0;
    unsaved->Contents = contents;

    /* Close the file. */
    fclose(to_file);

    /* Copy the file name that we're remapping from. */
    filename_len = semi - arg_string;
    filename = (char *)malloc(filename_len + 1);
    memcpy(filename, arg_string, filename_len);
    filename[filename_len] = 0;
    unsaved->Filename = filename;
  }

  return 0;
}

/******************************************************************************/
/* Pretty-printing.                                                           */
/******************************************************************************/

int want_display_name = 0;

static void PrintCursor(CXCursor Cursor) {
  if (clang_isInvalid(Cursor.kind)) {
    CXString ks = clang_getCursorKindSpelling(Cursor.kind);
    printf("Invalid Cursor => %s", clang_getCString(ks));
    clang_disposeString(ks);
  }
  else {
    CXString string, ks;
    CXCursor Referenced;
    unsigned line, column;
    CXCursor SpecializationOf;
    CXCursor *overridden;
    unsigned num_overridden;

    ks = clang_getCursorKindSpelling(Cursor.kind);
    string = want_display_name? clang_getCursorDisplayName(Cursor) 
                              : clang_getCursorSpelling(Cursor);
    printf("%s=%s", clang_getCString(ks),
                    clang_getCString(string));
    clang_disposeString(ks);
    clang_disposeString(string);

    Referenced = clang_getCursorReferenced(Cursor);
    if (!clang_equalCursors(Referenced, clang_getNullCursor())) {
      if (clang_getCursorKind(Referenced) == CXCursor_OverloadedDeclRef) {
        unsigned I, N = clang_getNumOverloadedDecls(Referenced);
        printf("[");
        for (I = 0; I != N; ++I) {
          CXCursor Ovl = clang_getOverloadedDecl(Referenced, I);
          CXSourceLocation Loc;
          if (I)
            printf(", ");
          
          Loc = clang_getCursorLocation(Ovl);
          clang_getSpellingLocation(Loc, 0, &line, &column, 0);
          printf("%d:%d", line, column);          
        }
        printf("]");
      } else {
        CXSourceLocation Loc = clang_getCursorLocation(Referenced);
        clang_getSpellingLocation(Loc, 0, &line, &column, 0);
        printf(":%d:%d", line, column);
      }
    }

    if (clang_isCursorDefinition(Cursor))
      printf(" (Definition)");
    
    switch (clang_getCursorAvailability(Cursor)) {
      case CXAvailability_Available:
        break;
        
      case CXAvailability_Deprecated:
        printf(" (deprecated)");
        break;
        
      case CXAvailability_NotAvailable:
        printf(" (unavailable)");
        break;
    }
    
    if (Cursor.kind == CXCursor_IBOutletCollectionAttr) {
      CXType T =
        clang_getCanonicalType(clang_getIBOutletCollectionType(Cursor));
      CXString S = clang_getTypeKindSpelling(T.kind);
      printf(" [IBOutletCollection=%s]", clang_getCString(S));
      clang_disposeString(S);
    }
    
    if (Cursor.kind == CXCursor_CXXBaseSpecifier) {
      enum CX_CXXAccessSpecifier access = clang_getCXXAccessSpecifier(Cursor);
      unsigned isVirtual = clang_isVirtualBase(Cursor);
      const char *accessStr = 0;

      switch (access) {
        case CX_CXXInvalidAccessSpecifier:
          accessStr = "invalid"; break;
        case CX_CXXPublic:
          accessStr = "public"; break;
        case CX_CXXProtected:
          accessStr = "protected"; break;
        case CX_CXXPrivate:
          accessStr = "private"; break;
      }      
      
      printf(" [access=%s isVirtual=%s]", accessStr,
             isVirtual ? "true" : "false");
    }
    
    SpecializationOf = clang_getSpecializedCursorTemplate(Cursor);
    if (!clang_equalCursors(SpecializationOf, clang_getNullCursor())) {
      CXSourceLocation Loc = clang_getCursorLocation(SpecializationOf);
      CXString Name = clang_getCursorSpelling(SpecializationOf);
      clang_getSpellingLocation(Loc, 0, &line, &column, 0);
      printf(" [Specialization of %s:%d:%d]", 
             clang_getCString(Name), line, column);
      clang_disposeString(Name);
    }

    clang_getOverriddenCursors(Cursor, &overridden, &num_overridden);
    if (num_overridden) {      
      unsigned I;
      printf(" [Overrides ");
      for (I = 0; I != num_overridden; ++I) {
        CXSourceLocation Loc = clang_getCursorLocation(overridden[I]);
        clang_getSpellingLocation(Loc, 0, &line, &column, 0);
        if (I)
          printf(", ");
        printf("@%d:%d", line, column);
      }
      printf("]");
      clang_disposeOverriddenCursors(overridden);
    }
    
    if (Cursor.kind == CXCursor_InclusionDirective) {
      CXFile File = clang_getIncludedFile(Cursor);
      CXString Included = clang_getFileName(File);
      printf(" (%s)", clang_getCString(Included));
      clang_disposeString(Included);
    }
  }
}

static const char* GetCursorSource(CXCursor Cursor) {
  CXSourceLocation Loc = clang_getCursorLocation(Cursor);
  CXString source;
  CXFile file;
  clang_getSpellingLocation(Loc, &file, 0, 0, 0);
  source = clang_getFileName(file);
  if (!clang_getCString(source)) {
    clang_disposeString(source);
    return "<invalid loc>";
  }
  else {
    const char *b = basename(clang_getCString(source));
    clang_disposeString(source);
    return b;
  }
}

/******************************************************************************/
/* Callbacks.                                                                 */
/******************************************************************************/

typedef void (*PostVisitTU)(CXTranslationUnit);

void PrintDiagnostic(CXDiagnostic Diagnostic) {
  FILE *out = stderr;
  CXFile file;
  CXString Msg;
  unsigned display_opts = CXDiagnostic_DisplaySourceLocation
    | CXDiagnostic_DisplayColumn | CXDiagnostic_DisplaySourceRanges
    | CXDiagnostic_DisplayOption;
  unsigned i, num_fixits;

  if (clang_getDiagnosticSeverity(Diagnostic) == CXDiagnostic_Ignored)
    return;

  Msg = clang_formatDiagnostic(Diagnostic, display_opts);
  fprintf(stderr, "%s\n", clang_getCString(Msg));
  clang_disposeString(Msg);

  clang_getSpellingLocation(clang_getDiagnosticLocation(Diagnostic),
                            &file, 0, 0, 0);
  if (!file)
    return;

  num_fixits = clang_getDiagnosticNumFixIts(Diagnostic);
  for (i = 0; i != num_fixits; ++i) {
    CXSourceRange range;
    CXString insertion_text = clang_getDiagnosticFixIt(Diagnostic, i, &range);
    CXSourceLocation start = clang_getRangeStart(range);
    CXSourceLocation end = clang_getRangeEnd(range);
    unsigned start_line, start_column, end_line, end_column;
    CXFile start_file, end_file;
    clang_getSpellingLocation(start, &start_file, &start_line,
                              &start_column, 0);
    clang_getSpellingLocation(end, &end_file, &end_line, &end_column, 0);
    if (clang_equalLocations(start, end)) {
      /* Insertion. */
      if (start_file == file)
        fprintf(out, "FIX-IT: Insert \"%s\" at %d:%d\n",
                clang_getCString(insertion_text), start_line, start_column);
    } else if (strcmp(clang_getCString(insertion_text), "") == 0) {
      /* Removal. */
      if (start_file == file && end_file == file) {
        fprintf(out, "FIX-IT: Remove ");
        PrintExtent(out, start_line, start_column, end_line, end_column);
        fprintf(out, "\n");
      }
    } else {
      /* Replacement. */
      if (start_file == end_file) {
        fprintf(out, "FIX-IT: Replace ");
        PrintExtent(out, start_line, start_column, end_line, end_column);
        fprintf(out, " with \"%s\"\n", clang_getCString(insertion_text));
      }
      break;
    }
    clang_disposeString(insertion_text);
  }
}

void PrintDiagnostics(CXTranslationUnit TU) {
  int i, n = clang_getNumDiagnostics(TU);
  for (i = 0; i != n; ++i) {
    CXDiagnostic Diag = clang_getDiagnostic(TU, i);
    PrintDiagnostic(Diag);
    clang_disposeDiagnostic(Diag);
  }
}

/******************************************************************************/
/* Logic for testing traversal.                                               */
/******************************************************************************/

static const char *FileCheckPrefix = "CHECK";

static void PrintCursorExtent(CXCursor C) {
  CXSourceRange extent = clang_getCursorExtent(C);
  CXFile begin_file, end_file;
  unsigned begin_line, begin_column, end_line, end_column;

  clang_getSpellingLocation(clang_getRangeStart(extent),
                            &begin_file, &begin_line, &begin_column, 0);
  clang_getSpellingLocation(clang_getRangeEnd(extent),
                            &end_file, &end_line, &end_column, 0);
  if (!begin_file || !end_file)
    return;

  printf(" Extent=");
  PrintExtent(stdout, begin_line, begin_column, end_line, end_column);
}

/* Data used by all of the visitors. */
typedef struct  {
  CXTranslationUnit TU;
  enum CXCursorKind *Filter;
} VisitorData;


enum CXChildVisitResult FilteredPrintingVisitor(CXCursor Cursor,
                                                CXCursor Parent,
                                                CXClientData ClientData) {
  VisitorData *Data = (VisitorData *)ClientData;
  if (!Data->Filter || (Cursor.kind == *(enum CXCursorKind *)Data->Filter)) {
    CXSourceLocation Loc = clang_getCursorLocation(Cursor);
    unsigned line, column;
    clang_getSpellingLocation(Loc, 0, &line, &column, 0);
    printf("// %s: %s:%d:%d: ", FileCheckPrefix,
           GetCursorSource(Cursor), line, column);
    PrintCursor(Cursor);
    PrintCursorExtent(Cursor);
    printf("\n");
    return CXChildVisit_Recurse;
  }

  return CXChildVisit_Continue;
}

static enum CXChildVisitResult FunctionScanVisitor(CXCursor Cursor,
                                                   CXCursor Parent,
                                                   CXClientData ClientData) {
  const char *startBuf, *endBuf;
  unsigned startLine, startColumn, endLine, endColumn, curLine, curColumn;
  CXCursor Ref;
  VisitorData *Data = (VisitorData *)ClientData;

  if (Cursor.kind != CXCursor_FunctionDecl ||
      !clang_isCursorDefinition(Cursor))
    return CXChildVisit_Continue;

  clang_getDefinitionSpellingAndExtent(Cursor, &startBuf, &endBuf,
                                       &startLine, &startColumn,
                                       &endLine, &endColumn);
  /* Probe the entire body, looking for both decls and refs. */
  curLine = startLine;
  curColumn = startColumn;

  while (startBuf < endBuf) {
    CXSourceLocation Loc;
    CXFile file;
    CXString source;

    if (*startBuf == '\n') {
      startBuf++;
      curLine++;
      curColumn = 1;
    } else if (*startBuf != '\t')
      curColumn++;

    Loc = clang_getCursorLocation(Cursor);
    clang_getSpellingLocation(Loc, &file, 0, 0, 0);

    source = clang_getFileName(file);
    if (clang_getCString(source)) {
      CXSourceLocation RefLoc
        = clang_getLocation(Data->TU, file, curLine, curColumn);
      Ref = clang_getCursor(Data->TU, RefLoc);
      if (Ref.kind == CXCursor_NoDeclFound) {
        /* Nothing found here; that's fine. */
      } else if (Ref.kind != CXCursor_FunctionDecl) {
        printf("// %s: %s:%d:%d: ", FileCheckPrefix, GetCursorSource(Ref),
               curLine, curColumn);
        PrintCursor(Ref);
        printf("\n");
      }
    }
    clang_disposeString(source);
    startBuf++;
  }

  return CXChildVisit_Continue;
}

/******************************************************************************/
/* USR testing.                                                               */
/******************************************************************************/

enum CXChildVisitResult USRVisitor(CXCursor C, CXCursor parent,
                                   CXClientData ClientData) {
  VisitorData *Data = (VisitorData *)ClientData;
  if (!Data->Filter || (C.kind == *(enum CXCursorKind *)Data->Filter)) {
    CXString USR = clang_getCursorUSR(C);
    const char *cstr = clang_getCString(USR);
    if (!cstr || cstr[0] == '\0') {
      clang_disposeString(USR);
      return CXChildVisit_Recurse;
    }
    printf("// %s: %s %s", FileCheckPrefix, GetCursorSource(C), cstr);

    PrintCursorExtent(C);
    printf("\n");
    clang_disposeString(USR);

    return CXChildVisit_Recurse;
  }

  return CXChildVisit_Continue;
}

/******************************************************************************/
/* Inclusion stack testing.                                                   */
/******************************************************************************/

void InclusionVisitor(CXFile includedFile, CXSourceLocation *includeStack,
                      unsigned includeStackLen, CXClientData data) {

  unsigned i;
  CXString fname;

  fname = clang_getFileName(includedFile);
  printf("file: %s\nincluded by:\n", clang_getCString(fname));
  clang_disposeString(fname);

  for (i = 0; i < includeStackLen; ++i) {
    CXFile includingFile;
    unsigned line, column;
    clang_getSpellingLocation(includeStack[i], &includingFile, &line,
                              &column, 0);
    fname = clang_getFileName(includingFile);
    printf("  %s:%d:%d\n", clang_getCString(fname), line, column);
    clang_disposeString(fname);
  }
  printf("\n");
}

void PrintInclusionStack(CXTranslationUnit TU) {
  clang_getInclusions(TU, InclusionVisitor, NULL);
}

/******************************************************************************/
/* Linkage testing.                                                           */
/******************************************************************************/

static enum CXChildVisitResult PrintLinkage(CXCursor cursor, CXCursor p,
                                            CXClientData d) {
  const char *linkage = 0;

  if (clang_isInvalid(clang_getCursorKind(cursor)))
    return CXChildVisit_Recurse;

  switch (clang_getCursorLinkage(cursor)) {
    case CXLinkage_Invalid: break;
    case CXLinkage_NoLinkage: linkage = "NoLinkage"; break;
    case CXLinkage_Internal: linkage = "Internal"; break;
    case CXLinkage_UniqueExternal: linkage = "UniqueExternal"; break;
    case CXLinkage_External: linkage = "External"; break;
  }

  if (linkage) {
    PrintCursor(cursor);
    printf("linkage=%s\n", linkage);
  }

  return CXChildVisit_Recurse;
}

/******************************************************************************/
/* Typekind testing.                                                          */
/******************************************************************************/

static enum CXChildVisitResult PrintTypeKind(CXCursor cursor, CXCursor p,
                                             CXClientData d) {

  if (!clang_isInvalid(clang_getCursorKind(cursor))) {
    CXType T = clang_getCursorType(cursor);
    CXString S = clang_getTypeKindSpelling(T.kind);
    PrintCursor(cursor);
    printf(" typekind=%s", clang_getCString(S));
    if (clang_isConstQualifiedType(T))
      printf(" const");
    if (clang_isVolatileQualifiedType(T))
      printf(" volatile");
    if (clang_isRestrictQualifiedType(T))
      printf(" restrict");
    clang_disposeString(S);
    /* Print the canonical type if it is different. */
    {
      CXType CT = clang_getCanonicalType(T);
      if (!clang_equalTypes(T, CT)) {
        CXString CS = clang_getTypeKindSpelling(CT.kind);
        printf(" [canonical=%s]", clang_getCString(CS));
        clang_disposeString(CS);
      }
    }
    /* Print the return type if it exists. */
    {
      CXType RT = clang_getCursorResultType(cursor);
      if (RT.kind != CXType_Invalid) {
        CXString RS = clang_getTypeKindSpelling(RT.kind);
        printf(" [result=%s]", clang_getCString(RS));
        clang_disposeString(RS);
      }
    }
    /* Print if this is a non-POD type. */
    printf(" [isPOD=%d]", clang_isPODType(T));

    printf("\n");
  }
  return CXChildVisit_Recurse;
}


/******************************************************************************/
/* Loading ASTs/source.                                                       */
/******************************************************************************/

static int perform_test_load(CXIndex Idx, CXTranslationUnit TU,
                             const char *filter, const char *prefix,
                             CXCursorVisitor Visitor,
                             PostVisitTU PV) {

  if (prefix)
    FileCheckPrefix = prefix;

  if (Visitor) {
    enum CXCursorKind K = CXCursor_NotImplemented;
    enum CXCursorKind *ck = &K;
    VisitorData Data;

    /* Perform some simple filtering. */
    if (!strcmp(filter, "all") || !strcmp(filter, "local")) ck = NULL;
    else if (!strcmp(filter, "all-display") || 
             !strcmp(filter, "local-display")) {
      ck = NULL;
      want_display_name = 1;
    }
    else if (!strcmp(filter, "none")) K = (enum CXCursorKind) ~0;
    else if (!strcmp(filter, "category")) K = CXCursor_ObjCCategoryDecl;
    else if (!strcmp(filter, "interface")) K = CXCursor_ObjCInterfaceDecl;
    else if (!strcmp(filter, "protocol")) K = CXCursor_ObjCProtocolDecl;
    else if (!strcmp(filter, "function")) K = CXCursor_FunctionDecl;
    else if (!strcmp(filter, "typedef")) K = CXCursor_TypedefDecl;
    else if (!strcmp(filter, "scan-function")) Visitor = FunctionScanVisitor;
    else {
      fprintf(stderr, "Unknown filter for -test-load-tu: %s\n", filter);
      return 1;
    }

    Data.TU = TU;
    Data.Filter = ck;
    clang_visitChildren(clang_getTranslationUnitCursor(TU), Visitor, &Data);
  }

  if (PV)
    PV(TU);

  PrintDiagnostics(TU);
  clang_disposeTranslationUnit(TU);
  return 0;
}

int perform_test_load_tu(const char *file, const char *filter,
                         const char *prefix, CXCursorVisitor Visitor,
                         PostVisitTU PV) {
  CXIndex Idx;
  CXTranslationUnit TU;
  int result;
  Idx = clang_createIndex(/* excludeDeclsFromPCH */
                          !strcmp(filter, "local") ? 1 : 0,
                          /* displayDiagnosics=*/1);

  if (!CreateTranslationUnit(Idx, file, &TU)) {
    clang_disposeIndex(Idx);
    return 1;
  }

  result = perform_test_load(Idx, TU, filter, prefix, Visitor, PV);
  clang_disposeIndex(Idx);
  return result;
}

int perform_test_load_source(int argc, const char **argv,
                             const char *filter, CXCursorVisitor Visitor,
                             PostVisitTU PV) {
  CXIndex Idx;
  CXTranslationUnit TU;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  int result;
  
  Idx = clang_createIndex(/* excludeDeclsFromPCH */
                          (!strcmp(filter, "local") || 
                           !strcmp(filter, "local-display"))? 1 : 0,
                          /* displayDiagnosics=*/0);

  if (parse_remapped_files(argc, argv, 0, &unsaved_files, &num_unsaved_files)) {
    clang_disposeIndex(Idx);
    return -1;
  }

  TU = clang_createTranslationUnitFromSourceFile(Idx, 0,
                                                 argc - num_unsaved_files,
                                                 argv + num_unsaved_files,
                                                 num_unsaved_files,
                                                 unsaved_files);
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    free_remapped_files(unsaved_files, num_unsaved_files);
    clang_disposeIndex(Idx);
    return 1;
  }

  result = perform_test_load(Idx, TU, filter, NULL, Visitor, PV);
  free_remapped_files(unsaved_files, num_unsaved_files);
  clang_disposeIndex(Idx);
  return result;
}

int perform_test_reparse_source(int argc, const char **argv, int trials,
                                const char *filter, CXCursorVisitor Visitor,
                                PostVisitTU PV) {
  CXIndex Idx;
  CXTranslationUnit TU;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  int result;
  int trial;
  
  Idx = clang_createIndex(/* excludeDeclsFromPCH */
                          !strcmp(filter, "local") ? 1 : 0,
                          /* displayDiagnosics=*/0);
  
  if (parse_remapped_files(argc, argv, 0, &unsaved_files, &num_unsaved_files)) {
    clang_disposeIndex(Idx);
    return -1;
  }
  
  /* Load the initial translation unit -- we do this without honoring remapped
   * files, so that we have a way to test results after changing the source. */
  TU = clang_parseTranslationUnit(Idx, 0,
                                  argv + num_unsaved_files,
                                  argc - num_unsaved_files,
                                  0, 0, getDefaultParsingOptions());
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    free_remapped_files(unsaved_files, num_unsaved_files);
    clang_disposeIndex(Idx);
    return 1;
  }
  
  for (trial = 0; trial < trials; ++trial) {
    if (clang_reparseTranslationUnit(TU, num_unsaved_files, unsaved_files,
                                     clang_defaultReparseOptions(TU))) {
      fprintf(stderr, "Unable to reparse translation unit!\n");
      clang_disposeTranslationUnit(TU);
      free_remapped_files(unsaved_files, num_unsaved_files);
      clang_disposeIndex(Idx);
      return -1;      
    }
  }
  
  result = perform_test_load(Idx, TU, filter, NULL, Visitor, PV);
  free_remapped_files(unsaved_files, num_unsaved_files);
  clang_disposeIndex(Idx);
  return result;
}

/******************************************************************************/
/* Logic for testing clang_getCursor().                                       */
/******************************************************************************/

static void print_cursor_file_scan(CXCursor cursor,
                                   unsigned start_line, unsigned start_col,
                                   unsigned end_line, unsigned end_col,
                                   const char *prefix) {
  printf("// %s: ", FileCheckPrefix);
  if (prefix)
    printf("-%s", prefix);
  PrintExtent(stdout, start_line, start_col, end_line, end_col);
  printf(" ");
  PrintCursor(cursor);
  printf("\n");
}

static int perform_file_scan(const char *ast_file, const char *source_file,
                             const char *prefix) {
  CXIndex Idx;
  CXTranslationUnit TU;
  FILE *fp;
  CXCursor prevCursor = clang_getNullCursor();
  CXFile file;
  unsigned line = 1, col = 1;
  unsigned start_line = 1, start_col = 1;

  if (!(Idx = clang_createIndex(/* excludeDeclsFromPCH */ 1,
                                /* displayDiagnosics=*/1))) {
    fprintf(stderr, "Could not create Index\n");
    return 1;
  }

  if (!CreateTranslationUnit(Idx, ast_file, &TU))
    return 1;

  if ((fp = fopen(source_file, "r")) == NULL) {
    fprintf(stderr, "Could not open '%s'\n", source_file);
    return 1;
  }

  file = clang_getFile(TU, source_file);
  for (;;) {
    CXCursor cursor;
    int c = fgetc(fp);

    if (c == '\n') {
      ++line;
      col = 1;
    } else
      ++col;

    /* Check the cursor at this position, and dump the previous one if we have
     * found something new.
     */
    cursor = clang_getCursor(TU, clang_getLocation(TU, file, line, col));
    if ((c == EOF || !clang_equalCursors(cursor, prevCursor)) &&
        prevCursor.kind != CXCursor_InvalidFile) {
      print_cursor_file_scan(prevCursor, start_line, start_col,
                             line, col, prefix);
      start_line = line;
      start_col = col;
    }
    if (c == EOF)
      break;

    prevCursor = cursor;
  }

  fclose(fp);
  clang_disposeTranslationUnit(TU);
  clang_disposeIndex(Idx);
  return 0;
}

/******************************************************************************/
/* Logic for testing clang code completion.                                   */
/******************************************************************************/

/* Parse file:line:column from the input string. Returns 0 on success, non-zero
   on failure. If successful, the pointer *filename will contain newly-allocated
   memory (that will be owned by the caller) to store the file name. */
int parse_file_line_column(const char *input, char **filename, unsigned *line,
                           unsigned *column, unsigned *second_line,
                           unsigned *second_column) {
  /* Find the second colon. */
  const char *last_colon = strrchr(input, ':');
  unsigned values[4], i;
  unsigned num_values = (second_line && second_column)? 4 : 2;

  char *endptr = 0;
  if (!last_colon || last_colon == input) {
    if (num_values == 4)
      fprintf(stderr, "could not parse filename:line:column:line:column in "
              "'%s'\n", input);
    else
      fprintf(stderr, "could not parse filename:line:column in '%s'\n", input);
    return 1;
  }

  for (i = 0; i != num_values; ++i) {
    const char *prev_colon;

    /* Parse the next line or column. */
    values[num_values - i - 1] = strtol(last_colon + 1, &endptr, 10);
    if (*endptr != 0 && *endptr != ':') {
      fprintf(stderr, "could not parse %s in '%s'\n",
              (i % 2 ? "column" : "line"), input);
      return 1;
    }

    if (i + 1 == num_values)
      break;

    /* Find the previous colon. */
    prev_colon = last_colon - 1;
    while (prev_colon != input && *prev_colon != ':')
      --prev_colon;
    if (prev_colon == input) {
      fprintf(stderr, "could not parse %s in '%s'\n",
              (i % 2 == 0? "column" : "line"), input);
      return 1;
    }

    last_colon = prev_colon;
  }

  *line = values[0];
  *column = values[1];

  if (second_line && second_column) {
    *second_line = values[2];
    *second_column = values[3];
  }

  /* Copy the file name. */
  *filename = (char*)malloc(last_colon - input + 1);
  memcpy(*filename, input, last_colon - input);
  (*filename)[last_colon - input] = 0;
  return 0;
}

const char *
clang_getCompletionChunkKindSpelling(enum CXCompletionChunkKind Kind) {
  switch (Kind) {
  case CXCompletionChunk_Optional: return "Optional";
  case CXCompletionChunk_TypedText: return "TypedText";
  case CXCompletionChunk_Text: return "Text";
  case CXCompletionChunk_Placeholder: return "Placeholder";
  case CXCompletionChunk_Informative: return "Informative";
  case CXCompletionChunk_CurrentParameter: return "CurrentParameter";
  case CXCompletionChunk_LeftParen: return "LeftParen";
  case CXCompletionChunk_RightParen: return "RightParen";
  case CXCompletionChunk_LeftBracket: return "LeftBracket";
  case CXCompletionChunk_RightBracket: return "RightBracket";
  case CXCompletionChunk_LeftBrace: return "LeftBrace";
  case CXCompletionChunk_RightBrace: return "RightBrace";
  case CXCompletionChunk_LeftAngle: return "LeftAngle";
  case CXCompletionChunk_RightAngle: return "RightAngle";
  case CXCompletionChunk_Comma: return "Comma";
  case CXCompletionChunk_ResultType: return "ResultType";
  case CXCompletionChunk_Colon: return "Colon";
  case CXCompletionChunk_SemiColon: return "SemiColon";
  case CXCompletionChunk_Equal: return "Equal";
  case CXCompletionChunk_HorizontalSpace: return "HorizontalSpace";
  case CXCompletionChunk_VerticalSpace: return "VerticalSpace";
  }

  return "Unknown";
}

void print_completion_string(CXCompletionString completion_string, FILE *file) {
  int I, N;

  N = clang_getNumCompletionChunks(completion_string);
  for (I = 0; I != N; ++I) {
    CXString text;
    const char *cstr;
    enum CXCompletionChunkKind Kind
      = clang_getCompletionChunkKind(completion_string, I);

    if (Kind == CXCompletionChunk_Optional) {
      fprintf(file, "{Optional ");
      print_completion_string(
                clang_getCompletionChunkCompletionString(completion_string, I),
                              file);
      fprintf(file, "}");
      continue;
    } 

    if (Kind == CXCompletionChunk_VerticalSpace) {
      fprintf(file, "{VerticalSpace  }");
      continue;
    }

    text = clang_getCompletionChunkText(completion_string, I);
    cstr = clang_getCString(text);
    fprintf(file, "{%s %s}",
            clang_getCompletionChunkKindSpelling(Kind),
            cstr ? cstr : "");
    clang_disposeString(text);
  }

}

void print_completion_result(CXCompletionResult *completion_result,
                             CXClientData client_data) {
  FILE *file = (FILE *)client_data;
  CXString ks = clang_getCursorKindSpelling(completion_result->CursorKind);

  fprintf(file, "%s:", clang_getCString(ks));
  clang_disposeString(ks);

  print_completion_string(completion_result->CompletionString, file);
  fprintf(file, " (%u)", 
          clang_getCompletionPriority(completion_result->CompletionString));
  switch (clang_getCompletionAvailability(completion_result->CompletionString)){
  case CXAvailability_Available:
    break;
    
  case CXAvailability_Deprecated:
    fprintf(file, " (deprecated)");
    break;
    
  case CXAvailability_NotAvailable:
    fprintf(file, " (unavailable)");
    break;
  }
  fprintf(file, "\n");
}

int my_stricmp(const char *s1, const char *s2) {
  while (*s1 && *s2) {
    int c1 = tolower((unsigned char)*s1), c2 = tolower((unsigned char)*s2);
    if (c1 < c2)
      return -1;
    else if (c1 > c2)
      return 1;
    
    ++s1;
    ++s2;
  }
  
  if (*s1)
    return 1;
  else if (*s2)
    return -1;
  return 0;
}

int perform_code_completion(int argc, const char **argv, int timing_only) {
  const char *input = argv[1];
  char *filename = 0;
  unsigned line;
  unsigned column;
  CXIndex CIdx;
  int errorCode;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  CXCodeCompleteResults *results = 0;
  CXTranslationUnit TU = 0;
  unsigned I, Repeats = 1;
  unsigned completionOptions = clang_defaultCodeCompleteOptions();
  
  if (getenv("CINDEXTEST_CODE_COMPLETE_PATTERNS"))
    completionOptions |= CXCodeComplete_IncludeCodePatterns;
  
  if (timing_only)
    input += strlen("-code-completion-timing=");
  else
    input += strlen("-code-completion-at=");

  if ((errorCode = parse_file_line_column(input, &filename, &line, &column,
                                          0, 0)))
    return errorCode;

  if (parse_remapped_files(argc, argv, 2, &unsaved_files, &num_unsaved_files))
    return -1;

  CIdx = clang_createIndex(0, 0);
  
  if (getenv("CINDEXTEST_EDITING"))
    Repeats = 5;
  
  TU = clang_parseTranslationUnit(CIdx, 0,
                                  argv + num_unsaved_files + 2,
                                  argc - num_unsaved_files - 2,
                                  0, 0, getDefaultParsingOptions());
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    return 1;
  }

  if (clang_reparseTranslationUnit(TU, 0, 0, clang_defaultReparseOptions(TU))) {
    fprintf(stderr, "Unable to reparse translation init!\n");
    return 1;
  }
  
  for (I = 0; I != Repeats; ++I) {
    results = clang_codeCompleteAt(TU, filename, line, column,
                                   unsaved_files, num_unsaved_files,
                                   completionOptions);
    if (!results) {
      fprintf(stderr, "Unable to perform code completion!\n");
      return 1;
    }
    if (I != Repeats-1)
      clang_disposeCodeCompleteResults(results);
  }

  if (results) {
    unsigned i, n = results->NumResults;
    if (!timing_only) {      
      /* Sort the code-completion results based on the typed text. */
      clang_sortCodeCompletionResults(results->Results, results->NumResults);

      for (i = 0; i != n; ++i)
        print_completion_result(results->Results + i, stdout);
    }
    n = clang_codeCompleteGetNumDiagnostics(results);
    for (i = 0; i != n; ++i) {
      CXDiagnostic diag = clang_codeCompleteGetDiagnostic(results, i);
      PrintDiagnostic(diag);
      clang_disposeDiagnostic(diag);
    }
    clang_disposeCodeCompleteResults(results);
  }
  clang_disposeTranslationUnit(TU);
  clang_disposeIndex(CIdx);
  free(filename);

  free_remapped_files(unsaved_files, num_unsaved_files);

  return 0;
}

typedef struct {
  char *filename;
  unsigned line;
  unsigned column;
} CursorSourceLocation;

int inspect_cursor_at(int argc, const char **argv) {
  CXIndex CIdx;
  int errorCode;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  CXTranslationUnit TU;
  CXCursor Cursor;
  CursorSourceLocation *Locations = 0;
  unsigned NumLocations = 0, Loc;
  unsigned Repeats = 1;
  unsigned I;
  
  /* Count the number of locations. */
  while (strstr(argv[NumLocations+1], "-cursor-at=") == argv[NumLocations+1])
    ++NumLocations;

  /* Parse the locations. */
  assert(NumLocations > 0 && "Unable to count locations?");
  Locations = (CursorSourceLocation *)malloc(
                                  NumLocations * sizeof(CursorSourceLocation));
  for (Loc = 0; Loc < NumLocations; ++Loc) {
    const char *input = argv[Loc + 1] + strlen("-cursor-at=");
    if ((errorCode = parse_file_line_column(input, &Locations[Loc].filename,
                                            &Locations[Loc].line,
                                            &Locations[Loc].column, 0, 0)))
      return errorCode;
  }

  if (parse_remapped_files(argc, argv, NumLocations + 1, &unsaved_files,
                           &num_unsaved_files))
    return -1;

  if (getenv("CINDEXTEST_EDITING"))
    Repeats = 5;

  /* Parse the translation unit. When we're testing clang_getCursor() after
     reparsing, don't remap unsaved files until the second parse. */
  CIdx = clang_createIndex(1, 1);
  TU = clang_parseTranslationUnit(CIdx, argv[argc - 1],
                                  argv + num_unsaved_files + 1 + NumLocations,
                                  argc - num_unsaved_files - 2 - NumLocations,
                                  unsaved_files,
                                  Repeats > 1? 0 : num_unsaved_files,
                                  getDefaultParsingOptions());
                                                 
  if (!TU) {
    fprintf(stderr, "unable to parse input\n");
    return -1;
  }

  for (I = 0; I != Repeats; ++I) {
    if (Repeats > 1 &&
        clang_reparseTranslationUnit(TU, num_unsaved_files, unsaved_files, 
                                     clang_defaultReparseOptions(TU))) {
      clang_disposeTranslationUnit(TU);
      return 1;
    }
    
    for (Loc = 0; Loc < NumLocations; ++Loc) {
      CXFile file = clang_getFile(TU, Locations[Loc].filename);
      if (!file)
        continue;

      Cursor = clang_getCursor(TU,
                               clang_getLocation(TU, file, Locations[Loc].line,
                                                 Locations[Loc].column));
      if (I + 1 == Repeats) {
        PrintCursor(Cursor);
        printf("\n");
        free(Locations[Loc].filename);
      }
    }
  }
  
  PrintDiagnostics(TU);
  clang_disposeTranslationUnit(TU);
  clang_disposeIndex(CIdx);
  free(Locations);
  free_remapped_files(unsaved_files, num_unsaved_files);
  return 0;
}

int perform_token_annotation(int argc, const char **argv) {
  const char *input = argv[1];
  char *filename = 0;
  unsigned line, second_line;
  unsigned column, second_column;
  CXIndex CIdx;
  CXTranslationUnit TU = 0;
  int errorCode;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  CXToken *tokens;
  unsigned num_tokens;
  CXSourceRange range;
  CXSourceLocation startLoc, endLoc;
  CXFile file = 0;
  CXCursor *cursors = 0;
  unsigned i;

  input += strlen("-test-annotate-tokens=");
  if ((errorCode = parse_file_line_column(input, &filename, &line, &column,
                                          &second_line, &second_column)))
    return errorCode;

  if (parse_remapped_files(argc, argv, 2, &unsaved_files, &num_unsaved_files))
    return -1;

  CIdx = clang_createIndex(0, 1);
  TU = clang_createTranslationUnitFromSourceFile(CIdx, argv[argc - 1],
                                                 argc - num_unsaved_files - 3,
                                                 argv + num_unsaved_files + 2,
                                                 num_unsaved_files,
                                                 unsaved_files);
  if (!TU) {
    fprintf(stderr, "unable to parse input\n");
    clang_disposeIndex(CIdx);
    free(filename);
    free_remapped_files(unsaved_files, num_unsaved_files);
    return -1;
  }
  errorCode = 0;

  file = clang_getFile(TU, filename);
  if (!file) {
    fprintf(stderr, "file %s is not in this translation unit\n", filename);
    errorCode = -1;
    goto teardown;
  }

  startLoc = clang_getLocation(TU, file, line, column);
  if (clang_equalLocations(clang_getNullLocation(), startLoc)) {
    fprintf(stderr, "invalid source location %s:%d:%d\n", filename, line,
            column);
    errorCode = -1;
    goto teardown;
  }

  endLoc = clang_getLocation(TU, file, second_line, second_column);
  if (clang_equalLocations(clang_getNullLocation(), endLoc)) {
    fprintf(stderr, "invalid source location %s:%d:%d\n", filename,
            second_line, second_column);
    errorCode = -1;
    goto teardown;
  }

  range = clang_getRange(startLoc, endLoc);
  clang_tokenize(TU, range, &tokens, &num_tokens);
  cursors = (CXCursor *)malloc(num_tokens * sizeof(CXCursor));
  clang_annotateTokens(TU, tokens, num_tokens, cursors);
  for (i = 0; i != num_tokens; ++i) {
    const char *kind = "<unknown>";
    CXString spelling = clang_getTokenSpelling(TU, tokens[i]);
    CXSourceRange extent = clang_getTokenExtent(TU, tokens[i]);
    unsigned start_line, start_column, end_line, end_column;

    switch (clang_getTokenKind(tokens[i])) {
    case CXToken_Punctuation: kind = "Punctuation"; break;
    case CXToken_Keyword: kind = "Keyword"; break;
    case CXToken_Identifier: kind = "Identifier"; break;
    case CXToken_Literal: kind = "Literal"; break;
    case CXToken_Comment: kind = "Comment"; break;
    }
    clang_getSpellingLocation(clang_getRangeStart(extent),
                              0, &start_line, &start_column, 0);
    clang_getSpellingLocation(clang_getRangeEnd(extent),
                              0, &end_line, &end_column, 0);
    printf("%s: \"%s\" ", kind, clang_getCString(spelling));
    PrintExtent(stdout, start_line, start_column, end_line, end_column);
    if (!clang_isInvalid(cursors[i].kind)) {
      printf(" ");
      PrintCursor(cursors[i]);
    }
    printf("\n");
  }
  free(cursors);
  clang_disposeTokens(TU, tokens, num_tokens);

 teardown:
  PrintDiagnostics(TU);
  clang_disposeTranslationUnit(TU);
  clang_disposeIndex(CIdx);
  free(filename);
  free_remapped_files(unsaved_files, num_unsaved_files);
  return errorCode;
}

/******************************************************************************/
/* USR printing.                                                              */
/******************************************************************************/

static int insufficient_usr(const char *kind, const char *usage) {
  fprintf(stderr, "USR for '%s' requires: %s\n", kind, usage);
  return 1;
}

static unsigned isUSR(const char *s) {
  return s[0] == 'c' && s[1] == ':';
}

static int not_usr(const char *s, const char *arg) {
  fprintf(stderr, "'%s' argument ('%s') is not a USR\n", s, arg);
  return 1;
}

static void print_usr(CXString usr) {
  const char *s = clang_getCString(usr);
  printf("%s\n", s);
  clang_disposeString(usr);
}

static void display_usrs() {
  fprintf(stderr, "-print-usrs options:\n"
        " ObjCCategory <class name> <category name>\n"
        " ObjCClass <class name>\n"
        " ObjCIvar <ivar name> <class USR>\n"
        " ObjCMethod <selector> [0=class method|1=instance method] "
            "<class USR>\n"
          " ObjCProperty <property name> <class USR>\n"
          " ObjCProtocol <protocol name>\n");
}

int print_usrs(const char **I, const char **E) {
  while (I != E) {
    const char *kind = *I;
    unsigned len = strlen(kind);
    switch (len) {
      case 8:
        if (memcmp(kind, "ObjCIvar", 8) == 0) {
          if (I + 2 >= E)
            return insufficient_usr(kind, "<ivar name> <class USR>");
          if (!isUSR(I[2]))
            return not_usr("<class USR>", I[2]);
          else {
            CXString x;
            x.data = (void*) I[2];
            x.private_flags = 0;
            print_usr(clang_constructUSR_ObjCIvar(I[1], x));
          }

          I += 3;
          continue;
        }
        break;
      case 9:
        if (memcmp(kind, "ObjCClass", 9) == 0) {
          if (I + 1 >= E)
            return insufficient_usr(kind, "<class name>");
          print_usr(clang_constructUSR_ObjCClass(I[1]));
          I += 2;
          continue;
        }
        break;
      case 10:
        if (memcmp(kind, "ObjCMethod", 10) == 0) {
          if (I + 3 >= E)
            return insufficient_usr(kind, "<method selector> "
                "[0=class method|1=instance method] <class USR>");
          if (!isUSR(I[3]))
            return not_usr("<class USR>", I[3]);
          else {
            CXString x;
            x.data = (void*) I[3];
            x.private_flags = 0;
            print_usr(clang_constructUSR_ObjCMethod(I[1], atoi(I[2]), x));
          }
          I += 4;
          continue;
        }
        break;
      case 12:
        if (memcmp(kind, "ObjCCategory", 12) == 0) {
          if (I + 2 >= E)
            return insufficient_usr(kind, "<class name> <category name>");
          print_usr(clang_constructUSR_ObjCCategory(I[1], I[2]));
          I += 3;
          continue;
        }
        if (memcmp(kind, "ObjCProtocol", 12) == 0) {
          if (I + 1 >= E)
            return insufficient_usr(kind, "<protocol name>");
          print_usr(clang_constructUSR_ObjCProtocol(I[1]));
          I += 2;
          continue;
        }
        if (memcmp(kind, "ObjCProperty", 12) == 0) {
          if (I + 2 >= E)
            return insufficient_usr(kind, "<property name> <class USR>");
          if (!isUSR(I[2]))
            return not_usr("<class USR>", I[2]);
          else {
            CXString x;
            x.data = (void*) I[2];
            x.private_flags = 0;
            print_usr(clang_constructUSR_ObjCProperty(I[1], x));
          }
          I += 3;
          continue;
        }
        break;
      default:
        break;
    }
    break;
  }

  if (I != E) {
    fprintf(stderr, "Invalid USR kind: %s\n", *I);
    display_usrs();
    return 1;
  }
  return 0;
}

int print_usrs_file(const char *file_name) {
  char line[2048];
  const char *args[128];
  unsigned numChars = 0;

  FILE *fp = fopen(file_name, "r");
  if (!fp) {
    fprintf(stderr, "error: cannot open '%s'\n", file_name);
    return 1;
  }

  /* This code is not really all that safe, but it works fine for testing. */
  while (!feof(fp)) {
    char c = fgetc(fp);
    if (c == '\n') {
      unsigned i = 0;
      const char *s = 0;

      if (numChars == 0)
        continue;

      line[numChars] = '\0';
      numChars = 0;

      if (line[0] == '/' && line[1] == '/')
        continue;

      s = strtok(line, " ");
      while (s) {
        args[i] = s;
        ++i;
        s = strtok(0, " ");
      }
      if (print_usrs(&args[0], &args[i]))
        return 1;
    }
    else
      line[numChars++] = c;
  }

  fclose(fp);
  return 0;
}

/******************************************************************************/
/* Command line processing.                                                   */
/******************************************************************************/
int write_pch_file(const char *filename, int argc, const char *argv[]) {
  CXIndex Idx;
  CXTranslationUnit TU;
  struct CXUnsavedFile *unsaved_files = 0;
  int num_unsaved_files = 0;
  
  Idx = clang_createIndex(/* excludeDeclsFromPCH */1, /* displayDiagnosics=*/1);
  
  if (parse_remapped_files(argc, argv, 0, &unsaved_files, &num_unsaved_files)) {
    clang_disposeIndex(Idx);
    return -1;
  }
  
  TU = clang_parseTranslationUnit(Idx, 0,
                                  argv + num_unsaved_files,
                                  argc - num_unsaved_files,
                                  unsaved_files,
                                  num_unsaved_files,
                                  CXTranslationUnit_Incomplete);
  if (!TU) {
    fprintf(stderr, "Unable to load translation unit!\n");
    free_remapped_files(unsaved_files, num_unsaved_files);
    clang_disposeIndex(Idx);
    return 1;
  }

  if (clang_saveTranslationUnit(TU, filename, clang_defaultSaveOptions(TU)))
    fprintf(stderr, "Unable to write PCH file %s\n", filename);
  clang_disposeTranslationUnit(TU);
  free_remapped_files(unsaved_files, num_unsaved_files);
  clang_disposeIndex(Idx);
  return 0;  
}

/******************************************************************************/
/* Command line processing.                                                   */
/******************************************************************************/

static CXCursorVisitor GetVisitor(const char *s) {
  if (s[0] == '\0')
    return FilteredPrintingVisitor;
  if (strcmp(s, "-usrs") == 0)
    return USRVisitor;
  return NULL;
}

static void print_usage(void) {
  fprintf(stderr,
    "usage: c-index-test -code-completion-at=<site> <compiler arguments>\n"
    "       c-index-test -code-completion-timing=<site> <compiler arguments>\n"
    "       c-index-test -cursor-at=<site> <compiler arguments>\n"
    "       c-index-test -test-file-scan <AST file> <source file> "
          "[FileCheck prefix]\n"
    "       c-index-test -test-load-tu <AST file> <symbol filter> "
          "[FileCheck prefix]\n"
    "       c-index-test -test-load-tu-usrs <AST file> <symbol filter> "
           "[FileCheck prefix]\n"
    "       c-index-test -test-load-source <symbol filter> {<args>}*\n");
  fprintf(stderr,
    "       c-index-test -test-load-source-reparse <trials> <symbol filter> "
    "          {<args>}*\n"
    "       c-index-test -test-load-source-usrs <symbol filter> {<args>}*\n"
    "       c-index-test -test-annotate-tokens=<range> {<args>}*\n"
    "       c-index-test -test-inclusion-stack-source {<args>}*\n"
    "       c-index-test -test-inclusion-stack-tu <AST file>\n"
    "       c-index-test -test-print-linkage-source {<args>}*\n"
    "       c-index-test -test-print-typekind {<args>}*\n"
    "       c-index-test -print-usr [<CursorKind> {<args>}]*\n");
  fprintf(stderr,
    "       c-index-test -print-usr-file <file>\n"
    "       c-index-test -write-pch <file> <compiler arguments>\n\n");
  fprintf(stderr,
    " <symbol filter> values:\n%s",
    "   all - load all symbols, including those from PCH\n"
    "   local - load all symbols except those in PCH\n"
    "   category - only load ObjC categories (non-PCH)\n"
    "   interface - only load ObjC interfaces (non-PCH)\n"
    "   protocol - only load ObjC protocols (non-PCH)\n"
    "   function - only load functions (non-PCH)\n"
    "   typedef - only load typdefs (non-PCH)\n"
    "   scan-function - scan function bodies (non-PCH)\n\n");
}

/***/

int cindextest_main(int argc, const char **argv) {
  clang_enableStackTraces();
  if (argc > 2 && strstr(argv[1], "-code-completion-at=") == argv[1])
    return perform_code_completion(argc, argv, 0);
  if (argc > 2 && strstr(argv[1], "-code-completion-timing=") == argv[1])
    return perform_code_completion(argc, argv, 1);
  if (argc > 2 && strstr(argv[1], "-cursor-at=") == argv[1])
    return inspect_cursor_at(argc, argv);
  else if (argc >= 4 && strncmp(argv[1], "-test-load-tu", 13) == 0) {
    CXCursorVisitor I = GetVisitor(argv[1] + 13);
    if (I)
      return perform_test_load_tu(argv[2], argv[3], argc >= 5 ? argv[4] : 0, I,
                                  NULL);
  }
  else if (argc >= 5 && strncmp(argv[1], "-test-load-source-reparse", 25) == 0){
    CXCursorVisitor I = GetVisitor(argv[1] + 25);
    if (I) {
      int trials = atoi(argv[2]);
      return perform_test_reparse_source(argc - 4, argv + 4, trials, argv[3], I, 
                                         NULL);
    }
  }
  else if (argc >= 4 && strncmp(argv[1], "-test-load-source", 17) == 0) {
    CXCursorVisitor I = GetVisitor(argv[1] + 17);
    if (I)
      return perform_test_load_source(argc - 3, argv + 3, argv[2], I, NULL);
  }
  else if (argc >= 4 && strcmp(argv[1], "-test-file-scan") == 0)
    return perform_file_scan(argv[2], argv[3],
                             argc >= 5 ? argv[4] : 0);
  else if (argc > 2 && strstr(argv[1], "-test-annotate-tokens=") == argv[1])
    return perform_token_annotation(argc, argv);
  else if (argc > 2 && strcmp(argv[1], "-test-inclusion-stack-source") == 0)
    return perform_test_load_source(argc - 2, argv + 2, "all", NULL,
                                    PrintInclusionStack);
  else if (argc > 2 && strcmp(argv[1], "-test-inclusion-stack-tu") == 0)
    return perform_test_load_tu(argv[2], "all", NULL, NULL,
                                PrintInclusionStack);
  else if (argc > 2 && strcmp(argv[1], "-test-print-linkage-source") == 0)
    return perform_test_load_source(argc - 2, argv + 2, "all", PrintLinkage,
                                    NULL);
  else if (argc > 2 && strcmp(argv[1], "-test-print-typekind") == 0)
    return perform_test_load_source(argc - 2, argv + 2, "all",
                                    PrintTypeKind, 0);
  else if (argc > 1 && strcmp(argv[1], "-print-usr") == 0) {
    if (argc > 2)
      return print_usrs(argv + 2, argv + argc);
    else {
      display_usrs();
      return 1;
    }
  }
  else if (argc > 2 && strcmp(argv[1], "-print-usr-file") == 0)
    return print_usrs_file(argv[2]);
  else if (argc > 2 && strcmp(argv[1], "-write-pch") == 0)
    return write_pch_file(argv[2], argc - 3, argv + 3);
           
  print_usage();
  return 1;
}

/***/

/* We intentionally run in a separate thread to ensure we at least minimal
 * testing of a multithreaded environment (for example, having a reduced stack
 * size). */

typedef struct thread_info {
  int argc;
  const char **argv;
  int result;
} thread_info;
void thread_runner(void *client_data_v) {
  thread_info *client_data = client_data_v;
  client_data->result = cindextest_main(client_data->argc, client_data->argv);
}

int main(int argc, const char **argv) {
  thread_info client_data;

  if (getenv("CINDEXTEST_NOTHREADS"))
    return cindextest_main(argc, argv);

  client_data.argc = argc;
  client_data.argv = argv;
  clang_executeOnThread(thread_runner, &client_data, 0);
  return client_data.result;
}
