// RUN: true

int strlen(char const *);
void puts(char const *);

struct String {
  long size;
  long capacity;
  char *storage;

  String() : size{0}, capacity{0}, storage{nullptr} {}
  String(char const *s) : size{strlen(s)}, capacity{size},
                          storage{new char[capacity]} {}
};

struct StringView {
  long size;
  char *storage;

  StringView(const String &s) : size{s.size}, storage{s.storage} {}
  StringView() : size{0}, storage{nullptr} {}
};

int main() {
  StringView sv;
  {
    String s = "Hi";
    sv = s;

    puts(sv.storage);
  }

  puts(sv.storage);
}
