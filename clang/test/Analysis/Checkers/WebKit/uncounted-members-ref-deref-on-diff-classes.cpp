// RUN: %clang_analyze_cc1 -analyzer-checker=webkit.NoUncountedMemberChecker -verify %s

#include "mock-types.h"

class RefCountedBase {
public:
  void ref() const { }
};

template<typename T> class RefCounted : public RefCountedBase {
public:
  virtual ~RefCounted() { }
  void deref() const { }
};

class TreeNode : public RefCounted<TreeNode> {
public:
  void setParent(TreeNode& parent) { m_parent = &parent; }

private:
  TreeNode* m_parent;
// expected-warning@-1{{Member variable 'm_parent' in 'TreeNode' is a raw pointer to ref-countable type 'TreeNode'}}
};
