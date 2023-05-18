add_subdirectory(testdata)
add_custom_target(libc_dirent_unittests)

add_libc_unittest(
  dirent_test
  SUITE
    libc_dirent_unittests
  SRCS
    dirent_test.cpp
  DEPENDS
    libc.src.__support.CPP.string_view
    libc.src.dirent.closedir
    libc.src.dirent.dirfd
    libc.src.dirent.opendir
    libc.src.dirent.readdir
    libc.src.errno.errno
)

