#pragma clang system_header

class String {
  char *storage{nullptr};
  long size;
  long capacity;

public:
  String() : size{0} {}
  String(int size) : size{size} {}
  String(const char *s) {}
};