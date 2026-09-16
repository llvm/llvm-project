// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s

struct Msg
{
  virtual ~Msg() {}
  virtual unsigned cmd() const = 0;
};

struct Ctrl : Msg
{
  unsigned c;
  unsigned cmd() const final { return c; }  // final: no override can exist
};

void clang_analyzer_dump(unsigned);
void clang_analyzer_eval(bool);

void test(Ctrl* p)
{
  clang_analyzer_dump(p->cmd());
  // expected-warning-re@-1 {{reg_${{[0-9]+}}<unsigned int Element{SymRegion{reg_${{[0-9]+}}<Ctrl * p>},0 S64b,struct Ctrl}.c>}}
  clang_analyzer_eval(p->cmd() == p->cmd()); // expected-warning {{TRUE}}
}
