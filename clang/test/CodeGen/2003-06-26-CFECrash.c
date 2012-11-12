// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

typedef struct min_info {
  long offset;
  unsigned file_attr;
} min_info;

typedef struct Globals {
  char answerbuf;
  min_info info[1];
  min_info *pInfo;
} Uz_Globs;

extern Uz_Globs G;

int extract_or_test_files() {  
  G.pInfo = G.info;
}

