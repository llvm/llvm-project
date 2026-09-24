namespace std {
inline namespace __cxx11 {
template <typename _CharT, typename = int, typename = _CharT>
class basic_string;
}
template <typename, typename> class basic_istream;
template <typename> struct __get_first_arg;
struct allocator_traits {
  using type = __get_first_arg<int>;
};
} // namespace std
namespace std {
inline namespace __cxx11 {
template <typename, typename, typename> class basic_string {
  allocator_traits _M_allocated_capacity;
  void _M_assign();
};
} // namespace __cxx11
} // namespace std
namespace std {
template <typename _CharT, typename _Alloc> void operator!=(_Alloc, _CharT);
template <typename _CharT, typename _Traits, typename _Alloc>
basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &,
                                        basic_string<_CharT, _Traits, _Alloc> &,
                                        _CharT);
} // namespace std
namespace std {
template <typename _CharT, typename _Traits, typename _Alloc>
void basic_string<_CharT, _Traits, _Alloc>::_M_assign() {
  this != 0;
}
template <typename _CharT, typename _Traits, typename _Alloc>
basic_istream<_CharT, _Traits> &getline(basic_istream<_CharT, _Traits> &,
                                        basic_string<_CharT, _Traits, _Alloc> &,
                                        _CharT) {}
} // namespace std
struct CommandLineOptionDefinition {
  void *OutAddress;
};
struct CommandLineCommand {
  CommandLineOptionDefinition Options;
};
namespace CommandLine {
extern const CommandLineCommand RootCommands[];
extern const int RootExamples[];
} // namespace CommandLine
using utf8 = char;
using u8string = std::basic_string<utf8>;
u8string _rct2DataPath;
CommandLineOptionDefinition StandardOptions{&_rct2DataPath};
const CommandLineCommand CommandLine::RootCommands[]{StandardOptions};
const int CommandLine::RootExamples[]{};
