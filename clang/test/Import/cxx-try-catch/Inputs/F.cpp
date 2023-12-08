void f() {
  try {
  } catch (...) {
  }

  try {
  } catch (int) {
  }

  try {
  } catch (int varname) {
  }

  try {
  } catch (int varname1) {
  } catch (long varname2) {
  }
}
