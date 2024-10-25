// RUN: %check_clang_tidy %s performance-avoid-endl %t

namespace std {
  template <typename CharT>
  class basic_ostream {
    public:
    template <typename T>
    basic_ostream& operator<<(T);
    basic_ostream& operator<<(basic_ostream<CharT>& (*)(basic_ostream<CharT>&));
  };

  template <typename CharT>
  class basic_iostream : public basic_ostream<CharT> {};

  using ostream = basic_ostream<char>;
  using wostream = basic_ostream<wchar_t>;

  using iostream = basic_iostream<char>;
  using wiostream = basic_iostream<wchar_t>;

  ostream cout;
  wostream wcout;

  ostream cerr;
  wostream wcerr;

  ostream clog;
  wostream wclog;

  template<typename CharT>
  basic_ostream<CharT>& endl(basic_ostream<CharT>&);
} // namespace std

void good() {
  std::cout << "Hello" << '\n';
  std::cout << "World\n";

  std::wcout << "Hello" << '\n';
  std::wcout << "World\n";

  std::cerr << "Hello" << '\n';
  std::cerr << "World\n";

  std::wcerr << "Hello" << '\n';
  std::wcerr << "World\n";

  std::clog << "Hello" << '\n';
  std::clog << "World\n";

  std::wclog << "Hello" << '\n';
  std::wclog << "World\n";
}

void bad() {
  std::cout << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cout << "World" << '\n';
  std::wcout << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcout << "World" << '\n';
  std::cerr << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cerr << "World" << '\n';
  std::wcerr << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcerr << "World" << '\n';
  std::clog << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::clog << "World" << '\n';
  std::wclog << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wclog << "World" << '\n';
}

void bad_single_argument() {
  std::cout << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cout << '\n';
  std::wcout << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcout << '\n';
  std::cerr << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cerr << '\n';
  std::wcerr << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcerr << '\n';
  std::clog << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::clog << '\n';
  std::wclog << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wclog << '\n';
}

void bad_multiple() {
  std::cout << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:51: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cout << "Hello" << '\n' << "World" << '\n';
  std::wcout << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:52: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcout << "Hello" << '\n' << "World" << '\n';
  std::cerr << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:51: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cerr << "Hello" << '\n' << "World" << '\n';
  std::wcerr << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:52: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcerr << "Hello" << '\n' << "World" << '\n';
  std::clog << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:51: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::clog << "Hello" << '\n' << "World" << '\n';
  std::wclog << "Hello" << std::endl << "World" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-MESSAGES: :[[@LINE-2]]:52: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wclog << "Hello" << '\n' << "World" << '\n';
}

void bad_function_call() {
  std::endl(std::cout);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cout << '\n';
  std::endl(std::cout << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cout << "Hi" << '\n';
  std::endl(std::wcout);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcout << '\n';
  std::endl(std::wcout << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcout << "Hi" << '\n';
  std::endl(std::cerr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cerr << '\n';
  std::endl(std::cerr << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::cerr << "Hi" << '\n';
  std::endl(std::wcerr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcerr << '\n';
  std::endl(std::wcerr << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wcerr << "Hi" << '\n';
  std::endl(std::clog);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::clog << '\n';
  std::endl(std::clog << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::clog << "Hi" << '\n';
  std::endl(std::wclog);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wclog << '\n';
  std::endl(std::wclog << "Hi");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: std::wclog << "Hi" << '\n';
}

void bad_user_stream() {
  std::iostream my_iostream;
  std::wiostream my_wiostream;
  std::ostream my_ostream;
  std::wostream my_wostream;

  my_iostream << "Hi" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_iostream << "Hi" << '\n';
  my_wiostream << "Hi" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_wiostream << "Hi" << '\n';
  my_ostream << "Hi" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_ostream << "Hi" << '\n';
  my_wostream << "Hi" << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_wostream << "Hi" << '\n';

  std::endl(my_iostream);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_iostream << '\n';
  std::endl(my_wiostream);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_wiostream << '\n';
  std::endl(my_ostream);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_ostream << '\n';
  std::endl(my_wostream);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_wostream << '\n';
}

using namespace std;
void bad_using_namespace_std() {
  cout << "Hello" << endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: do not use 'endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: cout << "Hello" << '\n';
  endl(cout);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: cout << '\n';
}

namespace my_prefix = std;
void bad_using_user_namespace() {
  my_prefix::cout << "Hello" << my_prefix::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: do not use 'my_prefix::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_prefix::cout << "Hello" << '\n';
  my_prefix::endl(my_prefix::cout);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: do not use 'my_prefix::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: my_prefix::cout << '\n';
}

struct CustomLogger {
  template <typename T>
  std::ostream& operator<<(T);
  std::ostream& operator<<(std::ostream& (*)(std::ostream&));
};

void bad_custom_stream() {
  CustomLogger logger;

  logger << std::endl;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
  // CHECK-FIXES: logger << '\n';
}

namespace gh107859 {

#define ENDL std::endl;

void bad_macro() {
  std::cout << ENDL;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: do not use 'std::endl' with streams; use '\n' instead [performance-avoid-endl]
}

} // namespace gh107859
