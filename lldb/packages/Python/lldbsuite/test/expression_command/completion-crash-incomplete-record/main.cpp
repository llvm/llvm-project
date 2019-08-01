int i;
struct F {
  int &r;
  F() : r(i) {}
};
template <class T> struct unique_ptr {
  F i;
  unique_ptr() : i() {//%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
}
};
int main() {unique_ptr<F> u; }
