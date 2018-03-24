struct __attribute__((trivial_abi)) S_Trivial {
  ~S_Trivial() {}
  int ivar = 10;
};

S_Trivial takeTrivial(S_Trivial inVal)
{
  S_Trivial ret_val = inVal;
  ret_val.ivar = 30;
  return ret_val;   // Set a breakpoint here
}

struct S_NotTrivial {
  ~S_NotTrivial() {}
  int ivar = 10;
};

S_NotTrivial takeNotTrivial(S_NotTrivial inVal)
{
  S_NotTrivial ret_val = inVal;
  ret_val.ivar = 30;
  return ret_val;   // Set a breakpoint here
}

int
main()
{
  S_Trivial inVal, outVal;
  outVal = takeTrivial(inVal);

  S_NotTrivial inNotVal, outNotVal;
  outNotVal = takeNotTrivial(outNotVal);

  return 0; // Set another for return value
}
