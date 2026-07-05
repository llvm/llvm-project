void clang_analyzer_warnIfReached();

class PlistCheckMacro {
public:
  PlistCheckMacro () { }
  void run() {
    clang_analyzer_warnIfReached();
  }
};
