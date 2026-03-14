template <class DataType> DataType values(DataType) { __builtin_va_list ValueArgs; return DataType(); }

template <class DataType>
class opt {
public:
  template <class Mods>
  opt(Mods) {}
};

