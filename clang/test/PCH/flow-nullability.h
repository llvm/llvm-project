// Header precompiled with the flow-sensitive nullability LangOpts. Used by
// flow-nullability.cpp and flow-nullability-mismatch.cpp.

struct Node {
  int value;
  Node *_Nullable next;
};

int *_Nullable getInt();
