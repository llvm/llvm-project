// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir/sub
// RUN: touch %t.dir/sub/file
// RUN: ln -s %t.dir/sub/file %t.dir/link
// RUN: %clangxx -O0 -g %s -o %t && %run %t %t.dir
// RUN: %clangxx -O0 -g -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %t.dir

// UNSUPPORTED: android

#include <assert.h>
#include <ftw.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

static char g_subdir[PATH_MAX];
static int ftw_visited;
static int inner_ftw_visited;

static int inner_ftw_cb(const char *fpath, const struct stat *sb,
                        int typeflag) {
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(strlen(fpath) > 0);
  if (S_ISDIR(sb->st_mode))
    assert(typeflag == FTW_D || typeflag == FTW_DP || typeflag == FTW_DNR);
  ++inner_ftw_visited;
  return 0;
}

static int ftw_cb(const char *fpath, const struct stat *sb, int typeflag) {
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(strlen(fpath) > 0);
  if (strcmp(fpath, g_subdir) == 0) {
    // Exercise nested ftw() call to verify thread-local callback restoration.
    inner_ftw_visited = 0;
    assert(ftw(g_subdir, inner_ftw_cb, 4) == 0);
    assert(inner_ftw_visited == 2);
  }
  if (S_ISDIR(sb->st_mode))
    assert(typeflag == FTW_D || typeflag == FTW_DP || typeflag == FTW_DNR);
  ++ftw_visited;
  return 0;
}

static int nftw_visited;

static int nftw_cb(const char *fpath, const struct stat *sb, int typeflag,
                   struct FTW *ftwbuf) {
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(ftwbuf != nullptr);
  assert(strlen(fpath) > 0);
  assert(ftwbuf->base >= 0);
  assert(ftwbuf->level >= 0);
  assert(fpath[ftwbuf->base] != '\0');
  if (S_ISDIR(sb->st_mode))
    assert(typeflag == FTW_D || typeflag == FTW_DP || typeflag == FTW_DNR);
  ++nftw_visited;
  return 0;
}

#if defined(__GLIBC__)
static int ftw64_visited;

static int ftw64_cb(const char *fpath, const struct stat64 *sb, int typeflag) {
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(strlen(fpath) > 0);
  if (S_ISDIR(sb->st_mode))
    assert(typeflag == FTW_D || typeflag == FTW_DP || typeflag == FTW_DNR);
  ++ftw64_visited;
  return 0;
}

static int nftw64_visited;

static int nftw64_cb(const char *fpath, const struct stat64 *sb, int typeflag,
                     struct FTW *ftwbuf) {
  assert(fpath != nullptr);
  assert(sb != nullptr);
  assert(ftwbuf != nullptr);
  assert(strlen(fpath) > 0);
  assert(ftwbuf->base >= 0);
  assert(ftwbuf->level >= 0);
  if (ftwbuf->level > 0 && typeflag == FTW_D)
    return FTW_SKIP_SUBTREE;
  ++nftw64_visited;
  return FTW_CONTINUE;
}
#endif

int main(int argc, char **argv) {
  assert(argc > 1);
  const char *tmpdir = argv[1];
  snprintf(g_subdir, sizeof(g_subdir), "%s/sub", tmpdir);

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
