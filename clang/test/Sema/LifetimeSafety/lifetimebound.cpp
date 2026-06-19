// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-all -verify %s


using size_t = decltype(sizeof(0));
extern "C" size_t strlen(const char *);

#define LIFETIMEBOUND [[clang::lifetimebound]]

struct View
{
    View(const char* data LIFETIMEBOUND)
        : mData(data)
        , mSize(strlen(data))
    {}

    const char* data() const {
        return mData;
    }

    size_t size() const {
        return mSize;
    }

private:
    const char* mData;
    size_t mSize;
};

void test() {
  char *c = new char[5]; //expected-warning {{allocated object does not live long enough}}
  View v(c); 
  delete[] c;  // expected-note {{freed here}}
  const char *c1 = v.data(); // expected-note {{later used here}}
  return;
}