// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null

struct _GIOChannel {
  int write_buf;
  char partial_write_buf[6];
  int d :1;
};

void g_io_channel_init (struct _GIOChannel *channel) {
  channel->partial_write_buf[0];
}

