int main() {
  []()
  { //%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
  }
  ();
  struct {
      void f()
      { //%self.dbg.GetCommandInterpreter().HandleCompletion("e ", len("e "), 0, -1, lldb.SBStringList())
      }
  } A;
}
