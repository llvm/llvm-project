// Header precompiled with the flow-sensitive nullability LangOpts. Used by
// nullability-safety.cpp and nullability-safety-mismatch.cpp.

struct Node {
  int value;
  Node *_Nullable next;
};

int *_Nullable getInt();
