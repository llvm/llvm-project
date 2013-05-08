// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: not cpp11-migrate %t.cpp --

int main(int argc, char** argv) {
  return 0;
}
