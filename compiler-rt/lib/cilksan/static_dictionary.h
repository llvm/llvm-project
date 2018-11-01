// -*- C++ -*-
#ifndef __STATIC_DICTIONARY__
#define __STATIC_DICTIONARY__

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <future>
#include <inttypes.h>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_map>

//#include "mem_access.h"
//#include "cilksan_internal.h"
//#include "debug_util.h"
#include "dictionary.h"
#include "disjointset.h"
#include "frame_data.h"
#include "spbag.h"
#include "stack.h"

#include <snappy.h>

/* Macros for memory pages in our dictionary. */

#define PAGE_SIZE_BITS 10
#define PAGE_SIZE ((uint64_t)(1ULL << PAGE_SIZE_BITS))
#define GET_PAGE_ID(key) (((uint64_t)(key)) >> PAGE_SIZE_BITS)
#define GET_PAGE_OFFSET(key) (((uint64_t)(key)) & (PAGE_SIZE-1))
// MAX_CACHE_SIZE amortizes the cost of compressing/decompressing pages.
// #define MAX_CACHE_SIZE 100
#define MAX_CACHE_SIZE 512
// MAX_CACHE_SCAN_LENGTH exploits locality to amortize the cost of looking up a
// page from the main structure.
#define MAX_CACHE_SCAN_LENGTH 4

class Static_Dictionary;

struct Page_t {
  bool is_compressed;
  uint64_t page_id;

  value_type00 *buffer;

  size_t compressed_size;
  unsigned char *compressed_buffer;
  std::future<void> fut;

  Page_t(uint64_t pg_id) {
    page_id = pg_id;
    is_compressed = false;
    buffer = new value_type00[PAGE_SIZE];
    // for (int i = 0; i < PAGE_SIZE; i++) {
    //     buffer[i] = (value_type00) 0ul;
    // }
    compressed_buffer = NULL;
  }

  ~Page_t() {
    decompress();

    for (size_t i = 0; i < PAGE_SIZE; i++) {
      // if (buffer[i].isValid())
      //   buffer[i].invalidate();
      // if (buffer[i] != 0) {
      //     buffer[i]->dec_ref_count();
      //     buffer[i] = 0;
      // }
    }

    delete[] buffer;
  }

  // TODO. Compress here. Should delete uncompressed_buffer and fill up compressed buffer
  // with the new compressed version. The LRU updates should be taken care of in a separate
  // function.
  void compress() {
    if (is_compressed)
      return;

    size_t in_len = PAGE_SIZE * sizeof(value_type00);
    char *out_buf =
      new char[snappy::MaxCompressedLength(in_len)];
    snappy::RawCompress((const char *)buffer, in_len,
                        out_buf, &compressed_size);

    /* check for an incompressible block */
    //printf("Compressed %d to %d bytes.\n", in_len, out_len);
    if (compressed_size >= in_len)
      printf("This block contains incompressible data.\n");

    compressed_buffer = new unsigned char[compressed_size];
    memcpy((void *)compressed_buffer, (void *)out_buf, compressed_size);

    delete[] out_buf;

    // We want to avoid invoking destructors on the elements in the buffer,
    // because these destructors affect reference counts.  (Conceputally,
    // the same references still exist, but in a compressed form.)  Hence,
    // we zero out the buffer before deleting it.
    memset((void *)buffer, 0, in_len);
    delete[] buffer;
    buffer = NULL;
    is_compressed = true;
    //std::cout << "done compressing" << std::endl;
  }

  // TODO: Same but decompress.
  void decompress() {
    // Get future state to wait for compression to finish.
    if (fut.valid())
      fut.get();

    if (!is_compressed)
      return;

    bool result;
    size_t uncompressed_length = sizeof(value_type00) * PAGE_SIZE;
    if (!snappy::GetUncompressedLength((const char *)compressed_buffer,
                                       compressed_size, &uncompressed_length))
      std::cerr << "Problem computing uncompressed length.\n";
    assert(uncompressed_length == sizeof(value_type00) * PAGE_SIZE &&
           "Uncompressed length does not match buffer size.");
    char *out_buf = new char[uncompressed_length];
    result = snappy::RawUncompress((const char *)compressed_buffer,
                                   compressed_size, out_buf);
    assert(result && "decompression failed");

    buffer = new value_type00[PAGE_SIZE];
    memcpy((void *)buffer, (void *)out_buf, uncompressed_length);

    delete[] out_buf;

    delete[] compressed_buffer;
    compressed_buffer = NULL;
    compressed_size = -1;
    is_compressed = false;
    //std::cout << "done decompressing" << std::endl;
  }
};

class LRU_Node {
public:
  Page_t *page;
  LRU_Node *next;
  LRU_Node *previous;

  LRU_Node() {
    next = NULL;
    page = NULL;
    previous = NULL;
  }

  LRU_Node(Page_t *p) {
    page = p;
    next = NULL;
    previous = NULL;
  }
};

class LRU_List {
public:
  LRU_Node free_list[MAX_CACHE_SIZE];
  LRU_Node *tail;
  LRU_Node *head;
  size_t cache_size;

  std::unordered_map<uint64_t, LRU_Node *> lru_page_ids;

  LRU_List();

  // Debugging
  void print_lru();
  void check_invariants(int label);

  void access(Page_t *page);
  Page_t *find_after_head(uint64_t page_id);
};

class Static_Dictionary : public Dictionary {
private:
  std::unordered_map<uint64_t, Page_t *> page_table;
  LRU_List lru_list;

public:
  Static_Dictionary();
  ~Static_Dictionary();

  void print_lru() {
    lru_list.print_lru();
  }

  value_type00 *find(uint64_t key);
  value_type00 *find_group(uint64_t key, size_t max_size, size_t &num_elems);
  value_type00 *find_exact_group(uint64_t key, size_t max_size,
                                 size_t &num_elems);

  const value_type00 &operator[] (uint64_t key);

  void erase(uint64_t key);
  void erase(uint64_t key, size_t size);
  bool includes(uint64_t key);
  bool includes(uint64_t key, size_t size);
  void insert(uint64_t key, const value_type00 &f);
  void insert(uint64_t key, size_t size, const value_type00 &f);
  void set(uint64_t key, size_t size, value_type00 &&f);
  void insert_into_found_group(uint64_t key, size_t size,
                               value_type00 *dst,
                               value_type00 &&f);
  void destruct();
};

#endif  // __STATIC_DICTIONARY__
