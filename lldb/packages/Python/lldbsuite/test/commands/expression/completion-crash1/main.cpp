namespace std {
struct a {
  a() {}
  a(a &&);
};
template <class> struct au {
  a ay;
  ~au() { //%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
  }
};
}
int main() { std::au<int>{}; }
