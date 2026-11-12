// RUN: %clangxx_msan -std=c++11 -O0 %s -o %t && %run %t
// RUN: %clangxx_msan -std=c++11 -fsanitize-memory-track-origins -O0 %s -o %t && %run %t

#include <assert.h>
#include <stdio.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  FILE *f = fopen(argv[0], "r");
  assert(f);
  char buf[50];
  fread(buf, 1, 1, f);
  fflush(f);

  assert(f->_IO_read_end > f->_IO_read_base);
  __msan_check_mem_is_initialized(f->_IO_read_end, f->_IO_read_end - f->_IO_read_base);

  char tmp_file[1000];
  sprintf(tmp_file, "%s.write.tmp", argv[0]);

  f = fopen(tmp_file, "w+");
  assert(f);
  fwrite(buf, 1, 1, f);
  fflush(f);

  assert(f->_IO_write_end > f->_IO_write_base);
  __msan_check_mem_is_initialized(f->_IO_write_end, f->_IO_write_end - f->_IO_write_base);
}
