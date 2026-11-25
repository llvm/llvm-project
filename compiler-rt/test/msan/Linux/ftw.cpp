// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/sub
// RUN: touch %t.dir/sub/file
// RUN: ln -s %t.dir/sub/file %t.dir/link
// RUN: %clangxx_msan -O0 %s -o %t && %run %t %t.dir
// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -O0 %s -o %t
// RUN: %run %t %t.dir
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %t.dir
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %t.dir

#include <assert.h>
#include <ftw.h>
#include <string.h>
#include <sys/stat.h>

#include <sanitizer/msan_interface.h>

static int ftw_visited;

static int ftw_cb(const char *fpath, const struct stat *sb, int typeflag) {
  __msan_check_mem_is_initialized(&fpath, sizeof(fpath));
  __msan_check_mem_is_initialized(&sb, sizeof(sb));
  __msan_check_mem_is_initialized(&typeflag, sizeof(typeflag));
  assert(fpath != nullptr);
  assert(sb != nullptr);
  __msan_check_mem_is_initialized(fpath, strlen(fpath) + 1);
  __msan_check_mem_is_initialized(sb, sizeof(*sb));
  ++ftw_visited;
  return 0;
}

static int nftw_visited;

static int nftw_cb(const char *fpath, const struct stat *sb, int typeflag,
                   struct FTW *ftwbuf) {
  __msan_check_mem_is_initialized(&fpath, sizeof(fpath));
  __msan_check_mem_is_initialized(&sb, sizeof(sb));
  __msan_check_mem_is_initialized(&typeflag, sizeof(typeflag));
  __msan_check_mem_is_initialized(&ftwbuf, sizeof(ftwbuf));
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(ftwbuf != nullptr);
  __msan_check_mem_is_initialized(fpath, strlen(fpath) + 1);
  __msan_check_mem_is_initialized(sb, sizeof(*sb));
  __msan_check_mem_is_initialized(ftwbuf, sizeof(*ftwbuf));
  struct stat st = *sb;
  if (S_ISDIR(st.st_mode))
    assert(typeflag == FTW_D || typeflag == FTW_DP || typeflag == FTW_DNR);
  assert(ftwbuf->base >= 0);
  assert(ftwbuf->level >= 0);
  ++nftw_visited;
  return 0;
}

#if defined(__GLIBC__)
static int ftw64_visited;

static int ftw64_cb(const char *fpath, const struct stat64 *sb, int typeflag) {
  __msan_check_mem_is_initialized(&fpath, sizeof(fpath));
  __msan_check_mem_is_initialized(&sb, sizeof(sb));
  __msan_check_mem_is_initialized(&typeflag, sizeof(typeflag));
  assert(fpath != nullptr);
  assert(sb != nullptr);
  __msan_check_mem_is_initialized(fpath, strlen(fpath) + 1);
  __msan_check_mem_is_initialized(sb, sizeof(*sb));
  ++ftw64_visited;
  return 0;
}

static int nftw64_visited;

static int nftw64_cb(const char *fpath, const struct stat64 *sb, int typeflag,
                     struct FTW *ftwbuf) {
  __msan_check_mem_is_initialized(&fpath, sizeof(fpath));
  __msan_check_mem_is_initialized(&sb, sizeof(sb));
  __msan_check_mem_is_initialized(&typeflag, sizeof(typeflag));
  __msan_check_mem_is_initialized(&ftwbuf, sizeof(ftwbuf));
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(ftwbuf != nullptr);
  __msan_check_mem_is_initialized(fpath, strlen(fpath) + 1);
  __msan_check_mem_is_initialized(sb, sizeof(*sb));
  __msan_check_mem_is_initialized(ftwbuf, sizeof(*ftwbuf));
  if (ftwbuf->level > 0 && typeflag == FTW_D)
    return FTW_SKIP_SUBTREE;
  ++nftw64_visited;
  return FTW_CONTINUE;
}
#endif

int main(int argc, char **argv) {
  assert(argc > 1);
  const char *tmpdir = argv[1];

  assert(ftw(tmpdir, ftw_cb, 16) == 0);
  assert(ftw_visited == 4);

  assert(nftw(tmpdir, nftw_cb, 16, FTW_PHYS) == 0);
  assert(nftw_visited == 4);

#if defined(__GLIBC__)
  assert(ftw64(tmpdir, ftw64_cb, 16) == 0);
  assert(ftw64_visited == 4);

  assert(nftw64(tmpdir, nftw64_cb, 16, FTW_PHYS | FTW_ACTIONRETVAL) == 0);
  // Root dir + link visited; sub returns FTW_SKIP_SUBTREE so sub/file is
  // skipped.
  assert(nftw64_visited == 2);
#endif

  return 0;
}
