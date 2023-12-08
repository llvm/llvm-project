template<typename _Tp>
class auto_ptr {
private:
  _Tp* _M_ptr;
public: 
  auto_ptr(_Tp* __p = 0) throw() : _M_ptr(__p) { }
  ~auto_ptr() { delete _M_ptr; }
};

void cause_div_by_zero_in_header(int in) {
  int h = 0;
  h = in/h;
  h++;
}

void do_something (int in) {
  in++;
  in++;
}

void cause_div_by_zero_in_header2(int in) {
  int h2 = 0;
  h2 = in/h2;
  h2++;
}

# define CALLS_BUGGY_FUNCTION2 cause_div_by_zero_in_header2(5);

void cause_div_by_zero_in_header3(int in) {
  int h3 = 0;
  h3 = in/h3;
  h3++;
}

# define CALLS_BUGGY_FUNCTION3 cause_div_by_zero_in_header3(5);

void cause_div_by_zero_in_header4(int in) {
  int h4 = 0;
  h4 = in/h4;
  h4++;
}

# define TAKE_CALL_AS_ARG(c) c;
