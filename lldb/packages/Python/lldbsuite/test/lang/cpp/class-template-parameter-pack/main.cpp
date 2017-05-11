//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LIDENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

template <class T, int... Args> struct C {
  T member;
  bool isSixteenThirtyTwo() { return false; }
};

template <> struct C<int, 16> {
  int member;
  bool isSixteenThirtyTwo() { return false; }
};

template <> struct C<int, 16, 32> : C<int, 16> {
  bool isSixteenThirtyTwo() { return true; }
};

template <class T, typename... Args> struct D {
  T member;
  bool isIntBool() { return false; }
};

template <> struct D<int, int> {
  int member;
  bool isIntBool() { return false; }
};

template <> struct D<int, int, bool> : D<int, int> {
  bool isIntBool() { return true; }
};

int main (int argc, char const *argv[])
{
    C<int,16,32> myC;
    C<int,16> myLesserC;
    myC.member = 64;
    (void)C<int,16,32>().isSixteenThirtyTwo();
    (void)C<int,16>().isSixteenThirtyTwo();
    (void)(myC.member != 64);   //% self.expect("expression -- myC", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["64"])
                                //% self.expect("expression -- C<int, 16>().isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
                                //% self.expect("expression -- C<int, 16, 32>().isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
                                //% self.expect("expression -- myLesserC.isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
                                //% self.expect("expression -- myC.isSixteenThirtyTwo()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
   
    D<int,int,bool> myD;
    D<int,int> myLesserD;
    myD.member = 64;
    (void)D<int,int,bool>().isIntBool();
    (void)D<int,int>().isIntBool();
    return myD.member != 64;	//% self.expect("expression -- myD", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["64"])
                                //% self.expect("expression -- D<int, int>().isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
                                //% self.expect("expression -- D<int, int, bool>().isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
                                //% self.expect("expression -- myLesserD.isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["false"])
                                //% self.expect("expression -- myD.isIntBool()", DATA_TYPES_DISPLAYED_CORRECTLY, substrs = ["true"])
}
