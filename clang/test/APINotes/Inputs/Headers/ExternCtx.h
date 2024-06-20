extern "C" {
  static int globalInExternC = 1;

  static void globalFuncInExternC() {}
}

extern "C++" {
  static int globalInExternCXX = 2;

  static void globalFuncInExternCXX() {}
}
