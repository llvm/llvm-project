typedef struct {
  id x;
} S;

id getObj(int c, id a) {
  // Commenting out the following line because AST importer crashes when trying
  // to import a BlockExpr.
  // return c ? ^{ return a; }() : ((S){ .x = a }).x;
  return ((S){ .x = a }).x;
}
