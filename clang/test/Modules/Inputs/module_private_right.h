__module_private__ double &f0(double);
__module_private__ double &f0(double);

__module_private__ int hidden_var;

inline void test_f0_in_right() {
  double &dr = f0(hidden_var);
}

struct VisibleStruct {
  __module_private__ int field;
  __module_private__ virtual void setField(int f);
};
