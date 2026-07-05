// RUN: %clang_cc1 -fsyntax-only -verify -Wformat-nonliteral -pedantic %s

#include <stdarg.h>

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
extern int vprintf(const char *restrict, va_list);
}

@class NSString;

@interface Format
+ (void)print:(NSString *)format, ... __attribute__((format(NSString, 1, 2)));
@end


namespace Templates {
  template<typename T>
  void my_uninstantiated_print(const T &arg) {
    [Format print:@"%d", arg];
  }

  template<typename T>
  void my_print(const T &arg) {
    [Format print:@"%d", arg]; // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
  }

  void use_my_print() {
    my_print("abc"); // expected-note {{requested here}}
  }


  template<typename T>
  class UninstantiatedPrinter {
  public:
    static void print(const T &arg) {
      [Format print:@"%d", arg]; // no-warning
    }
  };

  template<typename T>
  class Printer {
  public:
    void print(const T &arg) {
      [Format print:@"%d", arg]; // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
    }
  };

  void use_class(Printer<const char *> &p) {
    p.print("abc"); // expected-note {{requested here}}
  }


  template<typename T>
  class UninstantiatedWrapper {
  public:
    class Printer {
    public:
      void print(const T &arg) {
        [Format print:@"%d", arg]; // no-warning
      }
    };
  };

  template<typename T>
  class Wrapper {
  public:
    class Printer {
    public:
      void print(const T &arg) {
        [Format print:@"%d", arg]; // expected-warning {{format specifies type 'int' but the argument has type 'const char *'}}
      }
    };
  };

  void use_class(Wrapper<const char *>::Printer &p) {
    p.print("abc"); // expected-note {{requested here}}
  }
}

