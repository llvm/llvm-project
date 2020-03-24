#include "static_dictionary.h"
#include <sys/stat.h>

// Definitions of static members
const value_type00 Dictionary::null_val = value_type00();

int n_threads(void)
{
  struct stat task_stat;
  if (stat("/proc/self/task", &task_stat))
    return -1;

  return task_stat.st_nlink - 2;
}

LRU_List::LRU_List() {
  head = NULL;
  tail = NULL;
  cache_size = 0;

  for (int i = 0; i < MAX_CACHE_SIZE; i++) {
    free_list[i].page = NULL;
    free_list[i].next = NULL;
    free_list[i].previous = NULL;
  }
}

// Debugging
void LRU_List::print_lru() {
  std::cout << "Forward:" << std::endl;
  LRU_Node *current = head;
  std::cout << head << std::endl;
  while (current != tail) {
    std::cout << current->page->page_id << " -> ";
    current = current->next;
  }
  std::cout << current->page->page_id << std::endl;
  std::cout << "Backward:" << std::endl;
  current = tail;
  while (current != head) {
    std::cout << current->page->page_id << " -> ";
    current = current->previous;
  }
  std::cout << current->page->page_id << std::endl;
}

void LRU_List::check_invariants(int label) {
  LRU_Node *current = head;
  if (cache_size == 0) {
    assert(head == NULL && tail == NULL);
    return;
  }
  assert(head->previous == NULL);
  assert(tail->next == NULL);

  size_t counter = 1;
  while (current != tail) {
    assert(current->next);
    assert(current->next->previous == current);
    current = current->next;
    counter++;
    assert(counter <= MAX_CACHE_SIZE);
  }
  assert(lru_page_ids.size() == counter);
  assert(counter == cache_size && counter == lru_page_ids.size());
}

void LRU_List::access(Page_t *page) {
  // std::cerr << __PRETTY_FUNCTION__;
  // Check if in LRU. If it is, move to front and return.
  // If not then decompress, add to LRU, if max size then compress and remove last element.
  assert(cache_size <= MAX_CACHE_SIZE);
  uint64_t page_id = page->page_id;
  if (head && head->page->page_id == page_id)
    return;

  if (lru_page_ids.count(page_id) == 0) {
    // Decompress
    page->decompress();
    if (cache_size == MAX_CACHE_SIZE) {
      uint64_t page_to_remove = tail->page->page_id;
      tail->page->compress();
      tail->page = NULL;
      tail = tail->previous;
      tail->next = NULL;

      LRU_Node *recycled = lru_page_ids[page_to_remove];
      lru_page_ids.erase(page_to_remove);
      lru_page_ids.insert({page_id, recycled});

      recycled->page = page;
      recycled->next = head;
      recycled->previous = NULL;
      head = recycled;
      assert(head->next);
      head->next->previous = head;
    } else {
      LRU_Node *new_node = &(free_list[cache_size]);
      lru_page_ids[page_id] = new_node;
      new_node->page = page;
      new_node->next = head;
      new_node->previous = NULL;
      if (head == NULL)
        tail = new_node;

      head = new_node;
      if (head->next)
        head->next->previous = head;

      cache_size++;
    }
  } else {
    // Page found.
    LRU_Node *node = lru_page_ids[page_id];
    if (node->previous != NULL)
      node->previous->next = node->next;
    else
      return;  // Already first element.

    if (tail == node)
      tail = node->previous;


    if (node->next != NULL)
      node->next->previous = node->previous;

    node->next = head;
    node->previous = NULL;
    node->next->previous = node;
    head = node;
  }
}

Page_t *LRU_List::find_after_head(uint64_t page_id) {
  // Scan the list for the page
  if (!head)
    return nullptr;
  int count = 0;
  LRU_Node *node = head->next;
  while (__builtin_expect((node && count < MAX_CACHE_SCAN_LENGTH &&
                           node->page->page_id != page_id), 0)) {
    node = node->next;
    count++;
  }
  if (node && node->page->page_id == page_id) {
    assert(node->previous != nullptr);
    // Move the node to the front of the list.
    node->previous->next = node->next;
    if (tail == node)
      tail = node->previous;
    if (node->next != NULL)
      node->next->previous = node->previous;
    node->next = head;
    node->previous = NULL;
    node->next->previous = node;
    head = node;

    return node->page;
  }
  return nullptr;
}

Static_Dictionary::Static_Dictionary() {
}

// const value_type00 *
// Static_Dictionary::find_group(uint64_t key, size_t max_size, size_t &num_elems) {
//     uint64_t page_id = GET_PAGE_ID(key);
//     size_t my_num_elems = 1;
//     if (__builtin_expect((lru_list.head && lru_list.head->page->page_id == page_id), 1)) {
//         const value_type00 *acc = &lru_list.head->page->buffer[GET_PAGE_OFFSET(key)];
// #pragma unroll(4)
//         for (int i = 1; i < max_size; i++) {
//             const value_type00 *next = &acc[i];
//             if (__builtin_expect(next->getFunc() != acc->getFunc(), 0)) {
//               num_elems = my_num_elems;
//               return acc;
//             } else {
//               my_num_elems++;
//             }
//         }
//         num_elems = my_num_elems;
//         return acc;
//     }
//     auto found_page = page_table.find(page_id);
//     if (found_page != page_table.end()) {
//         lru_list.access(found_page->second);
//         const value_type00 *acc = &found_page->second->buffer[GET_PAGE_OFFSET(key)];
// #pragma unroll(4)
//         for (int i = 1; i < max_size; i++) {
//             const value_type00 *next = &acc[i];
//             if (__builtin_expect(next->getFunc() != acc->getFunc(), 0)) {
//               num_elems = my_num_elems;
//               return acc;
//             } else {
//               my_num_elems++;
//             }
//         }
//         num_elems = my_num_elems;
//         return acc;
//     } else {
//         num_elems = max_size;
//         return &null_val;
//     }
// }

value_type00 *
Static_Dictionary::find_group(uint64_t key, size_t max_size,
                              size_t &num_elems) {
  uint64_t page_id = GET_PAGE_ID(key);
  size_t my_num_elems = 1;
  value_type00 *acc = nullptr;
  if (__builtin_expect((lru_list.head &&
                        lru_list.head->page->page_id == page_id), 1)) {
    acc = &lru_list.head->page->buffer[GET_PAGE_OFFSET(key)];
  } else if (auto *page = lru_list.find_after_head(page_id)) {
    acc = &page->buffer[GET_PAGE_OFFSET(key)];
  } else {
    auto found_page = page_table.find(page_id);
    if (found_page != page_table.end()) {
      lru_list.access(found_page->second);
      acc = &found_page->second->buffer[GET_PAGE_OFFSET(key)];
    } else {
      num_elems = max_size;
      return nullptr;
    }
  }
  // #pragma unroll(8)
  for (size_t i = 1; i < max_size; i++) {
    const value_type00 *next = &acc[i];
    if (__builtin_expect(next->getFunc() == acc->getFunc(), 1)) {
      my_num_elems++;
    } else {
      break;
    }
  }
  num_elems = my_num_elems;
  return acc;
}

value_type00 *
Static_Dictionary::find_exact_group(uint64_t key, size_t max_size,
                                    size_t &num_elems) {
  uint64_t page_id = GET_PAGE_ID(key);
  size_t my_num_elems = 1;
  value_type00 *acc = nullptr;
  if (__builtin_expect((lru_list.head &&
                        lru_list.head->page->page_id == page_id), 1)) {
    acc = &lru_list.head->page->buffer[GET_PAGE_OFFSET(key)];
  } else if (auto *page = lru_list.find_after_head(page_id)) {
    acc = &page->buffer[GET_PAGE_OFFSET(key)];
  } else {
    auto found_page = page_table.find(page_id);
    if (found_page != page_table.end()) {
      lru_list.access(found_page->second);
      acc = &found_page->second->buffer[GET_PAGE_OFFSET(key)];
    } else {
      num_elems = max_size;
      return nullptr;
    }
  }
  // #pragma unroll(8)
  for (size_t i = 1; i < max_size; i++) {
    const value_type00 *next = &acc[i];
    if (__builtin_expect((next->getFunc() == acc->getFunc()) &&
                         next->sameAccessLocPtr(*acc), 1)) {
      my_num_elems++;
    } else {
      break;
    }
  }
  num_elems = my_num_elems;
  return acc;
}

value_type00 *Static_Dictionary::find(uint64_t key) {
  uint64_t page_id = GET_PAGE_ID(key);
  if (lru_list.head && lru_list.head->page->page_id == page_id) {
    return &lru_list.head->page->buffer[GET_PAGE_OFFSET(key)];
  }
  if (auto *page = lru_list.find_after_head(page_id)) {
    return &page->buffer[GET_PAGE_OFFSET(key)];
  }
  auto found_page = page_table.find(page_id);
  if (found_page != page_table.end()) {
    lru_list.access(found_page->second);
    return &found_page->second->buffer[GET_PAGE_OFFSET(key)];
  } else {
    return nullptr;
  }
}

const value_type00 &Static_Dictionary::operator[] (uint64_t key) {
  value_type00 *found = find(key);
  if (!found)
    return null_val;
  return *found;
}

/// Erase all entries from a region of this dictionary.  Not all entries in the
/// region need to be populated.
void Static_Dictionary::erase(uint64_t key, size_t size) {
  while (size > 0) {
    // Get the effective size: the size of the region to erase within the same
    // page.
    size_t eff_size = size;
    if (GET_PAGE_OFFSET(key) + size > PAGE_SIZE) {
      eff_size = (PAGE_SIZE - GET_PAGE_OFFSET(key));
    }
    assert(eff_size <= PAGE_SIZE);

    // Find the page containing the first part of the region starting at key.
    uint64_t page_id = GET_PAGE_ID(key);
    Page_t *page = nullptr;
    if (__builtin_expect((lru_list.head &&
                          lru_list.head->page->page_id == page_id), 1)) {
      page = lru_list.head->page;
    } else if ((page = lru_list.find_after_head(page_id))) {
    } else {
      auto found_page = page_table.find(page_id);
      if (found_page != page_table.end()) {
        // Decompress the page if need be.
        lru_list.access(found_page->second);
        page = found_page->second;
      } else {
      }
    }

    // If there's no page, move on to the next one.
    if (!page) {
      size -= eff_size;
      key += eff_size;
      continue;
    }

    // Update the access
    value_type00 *acc = &(page->buffer[GET_PAGE_OFFSET(key)]);
    if (acc) {
      // Update this access in groups of identical entries in the shadow memory.
      size_t group_start = 0;
      size_t group_size = 1;
      while (group_start + group_size < eff_size) {
        // Get a group of entries starting at acc that have identical
        // information.
        while (group_start + group_size < eff_size &&
               (acc[group_start + group_size].getFunc() ==
                acc[group_start].getFunc()) &&
               acc[group_start + group_size].sameAccessLocPtr(acc[group_start]))
          group_size++;
        // Clear the whole group.
        if (acc[group_start].isValid()) {
          acc[group_start].dec_ref_counts(group_size);
          for (size_t i = 0; i < group_size; ++i)
            acc[group_start + i].clear();
        }
        // Prepare to process the next group.
        group_start += group_size;
        group_size = 1;
      }
    }
    // Move on to the next page.
    size -= eff_size;
    key += eff_size;
  }
}

void Static_Dictionary::erase(uint64_t key) {
  assert(false);
}

bool Static_Dictionary::includes(uint64_t key, size_t size) {
  uint64_t page_id = GET_PAGE_ID(key);
  if (__builtin_expect((lru_list.head &&
                        lru_list.head->page->page_id == page_id), 1)) {
    const value_type00 *acc = &(lru_list.head->page->buffer[GET_PAGE_OFFSET(key)]);
    for (size_t i = 0; i < size; i++)
      if (acc[i].isValid())
        return true;
    return false;
  }
  if (auto *page = lru_list.find_after_head(page_id)) {
    const value_type00 *acc = &(page->buffer[GET_PAGE_OFFSET(key)]);
    for (size_t i = 0; i < size; i++)
      if (acc[i].isValid())
        return true;
    return false;
  }
  auto found_page = page_table.find(page_id);
  if (found_page != page_table.end()) {
    lru_list.access(found_page->second);
    const value_type00 *acc = &(found_page->second->buffer[GET_PAGE_OFFSET(key)]);
    for (size_t i = 0; i < size; i++)
      if (acc[i].isValid())
        return true;
    return false;
  } else {
    return false;
  }
}

bool Static_Dictionary::includes(uint64_t key) {
  uint64_t page_id = GET_PAGE_ID(key);
  if (lru_list.head && lru_list.head->page->page_id == page_id) {
    return lru_list.head->page->buffer[GET_PAGE_OFFSET(key)].isValid();
  }
  if (auto *page = lru_list.find_after_head(page_id)) {
    return page->buffer[GET_PAGE_OFFSET(key)].isValid();
  }
  auto found_page = page_table.find(page_id);
  if (found_page != page_table.end()) {
    lru_list.access(found_page->second);
    return found_page->second->buffer[GET_PAGE_OFFSET(key)].isValid();
  } else {
    return false;
  }
}

void Static_Dictionary::insert(uint64_t key, size_t size, const value_type00 &f) {
  value_type00 tmp(f);
  tmp.inc_ref_counts(size);
  uint64_t page_id = GET_PAGE_ID(key);
  if (lru_list.head && lru_list.head->page->page_id == page_id) {
    for (size_t i = 0; i < size; i++) {
      lru_list.head->page->buffer[GET_PAGE_OFFSET(key) + i].inherit(tmp);
    }
    return;
  }
  if (auto *page = lru_list.find_after_head(page_id)) {
    for (size_t i = 0; i < size; i++) {
      page->buffer[GET_PAGE_OFFSET(key) + i].inherit(tmp);
    }
    return;
  }
  auto found_page = page_table.find(page_id);
  if (found_page != page_table.end()) {
    lru_list.access(found_page->second);
    for (size_t i = 0; i < size; i++)
      found_page->second->buffer[GET_PAGE_OFFSET(key) + i].inherit(tmp);

  } else {
    Page_t *page = new Page_t(page_id);
    page_table[page_id] = page;
    lru_list.access(page);
    for (size_t i = 0; i < size; i++) {
      page->buffer[GET_PAGE_OFFSET(key) + i].inherit(tmp);
    }
  }
}

void Static_Dictionary::insert(uint64_t key, const value_type00 &f) {
  uint64_t page_id = GET_PAGE_ID(key);
  if (lru_list.head && lru_list.head->page->page_id == page_id) {
    lru_list.head->page->buffer[GET_PAGE_OFFSET(key)] = f;
    return;
  }
  if (auto *page = lru_list.find_after_head(page_id)) {
    page->buffer[GET_PAGE_OFFSET(key)] = f;
    return;
  }
  auto found_page = page_table.find(page_id);
  if (found_page != page_table.end()) {
    lru_list.access(found_page->second);
    found_page->second->buffer[GET_PAGE_OFFSET(key)] = f;
  } else {
    Page_t *page = new Page_t(page_id);
    page_table[page_id] = page;
    lru_list.access(page);
    page->buffer[GET_PAGE_OFFSET(key)] = f;
  }
}

void Static_Dictionary::set(uint64_t key, size_t size, value_type00 &&f) {
  f.inc_ref_counts(size);
  while (size > 0) {
    size_t eff_size = size;
    if (GET_PAGE_OFFSET(key) + size > PAGE_SIZE)
      eff_size = PAGE_SIZE - GET_PAGE_OFFSET(key);

    uint64_t page_id = GET_PAGE_ID(key);
    value_type00 *acc = nullptr;
    if (__builtin_expect((lru_list.head &&
                          lru_list.head->page->page_id == page_id), 1)) {
      acc = &(lru_list.head->page->buffer[GET_PAGE_OFFSET(key)]);
    } else if (auto *page = lru_list.find_after_head(page_id)) {
      acc = &(page->buffer[GET_PAGE_OFFSET(key)]);
    } else {
      auto found_page = page_table.find(page_id);
      if (__builtin_expect(found_page != page_table.end(), 1)) {
        lru_list.access(found_page->second);
        acc = &(found_page->second->buffer[GET_PAGE_OFFSET(key)]);
      } else {
        Page_t *page = new Page_t(page_id);
        page_table[page_id] = page;
        lru_list.access(page);
        acc = &(page->buffer[GET_PAGE_OFFSET(key)]);
        for (size_t i = 0; i < eff_size; ++i)
          acc[i].overwrite(f);
        acc = nullptr;
      }
    }
    if (acc) {
      size_t group_start = 0;
      size_t group_size = 1;
      while (group_start + group_size < eff_size) {
        while (group_start + group_size < eff_size &&
               (acc[group_start + group_size].getFunc() ==
                acc[group_start].getFunc()) &&
               acc[group_start + group_size].sameAccessLocPtr(acc[group_start]))
          group_size++;
        if (acc[group_start].isValid())
          acc[group_start].dec_ref_counts(group_size);

        group_start += group_size;
        group_size = 1;
      }
      for (size_t i = 0; i < eff_size; ++i)
        acc[i].overwrite(f);
    }

    size -= eff_size;
    key += eff_size;
  }
}

void Static_Dictionary::insert_into_found_group(uint64_t key, size_t size,
                                                value_type00 *dst,
                                                value_type00 &&f) {
  f.inc_ref_counts(size);
  uint64_t page_id = GET_PAGE_ID(key);
  assert(lru_list.head && lru_list.head->page->page_id == page_id);
  assert(dst == &lru_list.head->page->buffer[GET_PAGE_OFFSET(key)]);
  value_type00 *my_dst = dst;
  // Overwrite the table entries.
  for (size_t i = 0; i < size; i++) {
    my_dst[i].overwrite(f);
  }
}

Static_Dictionary::~Static_Dictionary() {
  for (auto iter = page_table.begin(); iter != page_table.end(); iter++) {
    delete iter->second;
  }
  page_table.clear();
  lru_list.lru_page_ids.clear();
  lru_list.head = NULL;
  lru_list.tail = NULL;
}
