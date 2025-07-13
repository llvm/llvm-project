#pragma once

#include <memory>
#include <optional>
#include <vector>

template <typename T> class NaryTree {
  struct Node {
    Node(const T &t) : value(t) {}

    std::vector<std::unique_ptr<Node>> children;
    T value;
  };

  using Element = std::unique_ptr<Node>;

  Element root{nullptr};

  template <typename Func>
  void traverse_post_order_impl(Element *parent, Element &current, int depth,
                                bool is_leaf, Func &&f) {
    for (auto &child : current->children)
      traverse_post_order_impl(&current, child, depth + 1,
                               !child->children.size(), f);

    f(parent, current, depth, !current->children.size());
  }

  template <typename Func>
  void traverse_pre_order_impl(Element *parent, Element &current, int depth,
                               bool is_leaf, Func &&f) {
    f(parent, current, depth, !current->children.size());

    for (auto &child : current->children)
      traverse_pre_order_impl(&current, child, depth + 1,
                              !child->children.size(), f);
  }

  const Element *find_node(const Element &current, const T &data) const {
    if (current) {
      if (current->value == data)
        return &current;
      else
        for (const auto &child : current->children)
          return find_node(child, data);
    }

    return nullptr;
  }

public:
  struct TraverseResult {
    std::optional<T> parent;
    T &self;
    int depth;
    bool is_leaf;
  };

  template <typename Func> void traverse_post_order(Func &&f) {
    traverse_post_order_impl(
        nullptr, root, 0,
        [&f](Element *parent, Element &n, int depth, bool is_leaf) {
          std::optional<T> opt;

          if (parent)
            opt = (*parent)->value;

          f(TraverseResult{opt, n->value, depth, is_leaf});
        });
  }

  template <typename Func> void traverse_pre_order(Func &&f) {
    traverse_pre_order_impl(
        nullptr, root, 0, !root->children.size(),
        [&f](Element *parent, Element &n, int depth, bool is_leaf) {
          std::optional<T> opt;

          if (parent)
            opt = (*parent)->value;

          f(TraverseResult{opt, n->value, depth, is_leaf});
        });
  }

  bool contains(const T &data) const {
    return find_node(root, data) != nullptr;
  }

  bool add_node(const T &parentData, const T &data) {
    if (!root) {
      root = std::make_unique<Node>(data);
      return true;
    }

    if (auto node = find_node(root, parentData)) {
      (*node)->children.push_back(std::make_unique<Node>(data));
      return true;
    }

    return false;
  }
};
