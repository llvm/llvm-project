// RUN: %clang_cc1 -triple x86_64-windows -fsyntax-only -verify -fborland-extensions -fcxx-exceptions %s

// This test is from http://docwiki.embarcadero.com/RADStudio/en/Try

int puts(const char *);

template<typename T>
int printf(const char *, T);

const char * strdup(const char *);

void free(const void *);

#define EXCEPTION_EXECUTE_HANDLER 1

class Exception
{
public:
  Exception(const char* s = "Unknown"){what = strdup(s);      }
  Exception(const Exception& e ){what = strdup(e.what); }
  ~Exception()                   {free(what);         }
  const char* msg() const             {return what;           }
private:
  const char* what;
};

int main()
{
  float e, f, g;
  try
  {
    try
    {
      f = 1.0;
      g = 0.0;
      try
      {
        puts("Another exception:");

        e = f / g;
      }
      __except(EXCEPTION_EXECUTE_HANDLER)
      {
        puts("Caught a C-based exception.");
        throw(Exception("Hardware error: Divide by 0"));
      }
    }
    catch(const Exception& e)
    {
      printf("Caught C++ Exception: %s :\n", e.msg());
    }
  }
  __finally
  {
    puts("C++ allows __finally too!");
  }
  return e;
}

namespace PR17584 {
template <typename>
void Except() {
  __try {
  } __except(true) {
  }
}

template <typename>
void Finally() {
  __try {
  } __finally {
  }
}

template void Except<void>();
template void Finally<void>();

}

void test___leave() {
  // Most tests are in __try.c.

  // Clang accepts try with __finally. MSVC doesn't. (Maybe a Borland thing?)
  // __leave in mixed blocks isn't supported.
  try {
    __leave; // expected-error{{'__leave' statement not in __try block}}
  } __finally {
  }
}
