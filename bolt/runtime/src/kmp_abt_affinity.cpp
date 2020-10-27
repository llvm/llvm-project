
/*
 * kmp_abt_affinity.cpp -- affinity parser for BOLT
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct kmp_abt_affinity_place {
  size_t num_ranks;
  int *ranks;
} kmp_abt_affinity_place_t;

typedef struct kmp_abt_affinity_places {
  size_t num_places;
  kmp_abt_affinity_place_t **p_places;
} kmp_abt_affinity_places_t;

typedef enum parse_pinterval {
  pinterval_pls,
  pinterval_pl,
  pinterval_p,
  pinterval_ip,
} parse_pinterval_t;

typedef enum parse_res_interval {
  res_interval_rns,
  res_interval_rn,
  res_interval_r,
  res_interval_ir,
} parse_res_interval_t;

typedef enum parse_word {
  word_sockets,
  word_cores,
  word_threads,
} parse_word_t;

typedef enum parse_num {
  num_any,
  num_positive,
  num_nonnegative,
} parse_num_t;

static kmp_abt_affinity_place_t *__kmp_abt_affinity_place_create() {
  kmp_abt_affinity_place_t *p_new_place = (kmp_abt_affinity_place_t *)
      malloc(sizeof(kmp_abt_affinity_place_t));
  p_new_place->num_ranks = 0;
  p_new_place->ranks = NULL;
  return p_new_place;
}

static kmp_abt_affinity_place_t *__kmp_abt_affinity_place_create_rank
    (int rank) {
  kmp_abt_affinity_place_t *p_new_place = __kmp_abt_affinity_place_create();
  p_new_place->num_ranks = 1;
  p_new_place->ranks = (int *)malloc(sizeof(int) * 1);
  p_new_place->ranks[0] = rank;
  return p_new_place;
}

static kmp_abt_affinity_place_t *__kmp_abt_affinity_place_create_place
    (const kmp_abt_affinity_place_t *p_place) {
  kmp_abt_affinity_place_t *p_new_place = __kmp_abt_affinity_place_create();
  size_t num_ranks = p_place->num_ranks;
  p_new_place->num_ranks = num_ranks;
  p_new_place->ranks = (int *)malloc(sizeof(int) * num_ranks);
  memcpy(p_new_place->ranks, p_place->ranks, sizeof(int) * num_ranks);
  return p_new_place;
}

static void __kmp_abt_affinity_place_free(kmp_abt_affinity_place_t *p_place) {
  free(p_place->ranks);
  free(p_place);
}

int __kmp_abt_affinity_place_find(const kmp_abt_affinity_place_t *p_place,
                                  int rank) {
  for (size_t i = 0, num_ranks = p_place->num_ranks; i < num_ranks; i++)
    if (p_place->ranks[i] == rank)
      return 1;
  return 0;
}

static void __kmp_abt_affinity_place_insert(kmp_abt_affinity_place_t *p_place,
                                            int rank) {
  if (__kmp_abt_affinity_place_find(p_place, rank))
    return;
  size_t num_ranks = p_place->num_ranks;
  size_t new_num_ranks = num_ranks + 1;
  int *new_ranks = (int *)malloc(sizeof(int) * new_num_ranks);
  memcpy(new_ranks, p_place->ranks, sizeof(int) * num_ranks);
  free(p_place->ranks);
  new_ranks[num_ranks] = rank;
  p_place->ranks = new_ranks;
  p_place->num_ranks = new_num_ranks;
}

static void __kmp_abt_affinity_place_insert_place
    (kmp_abt_affinity_place_t *p_place,
     const kmp_abt_affinity_place_t *p_inserted) {
  for (int i = 0, num_ranks = p_inserted->num_ranks; i < num_ranks; i++)
    __kmp_abt_affinity_place_insert(p_place, p_inserted->ranks[i]);
}

static kmp_abt_affinity_places_t *__kmp_abt_affinity_places_create() {
  kmp_abt_affinity_places_t *p_new_places = (kmp_abt_affinity_places_t *)
      malloc(sizeof(kmp_abt_affinity_places_t));
  p_new_places->num_places = 0;
  p_new_places->p_places = NULL;
  return p_new_places;
}

void __kmp_abt_affinity_places_free(kmp_abt_affinity_places_t *p_places) {
  for (size_t i = 0; i < p_places->num_places; i++)
    __kmp_abt_affinity_place_free(p_places->p_places[i]);
  free(p_places->p_places);
  free(p_places);
}

static void __kmp_abt_affinity_places_add(kmp_abt_affinity_places_t *p_places,
                                          kmp_abt_affinity_place_t *p_place) {
  size_t num_places = p_places->num_places;
  size_t new_num_places = num_places + 1;
  kmp_abt_affinity_place_t **p_new_places = (kmp_abt_affinity_place_t **)
      malloc(sizeof(kmp_abt_affinity_place_t *) * new_num_places);
  memcpy(p_new_places, p_places->p_places, sizeof(kmp_abt_affinity_place_t *)
                                           * num_places);
  free(p_places->p_places);
  p_new_places[num_places] = p_place;
  p_places->p_places = p_new_places;
  p_places->num_places = new_num_places;
}

static bool __kmp_abt_parse_num(int *p_val, parse_num_t type,
                                const char *str, size_t len,
                                size_t *p_consume) {
  size_t consume = 0;
  int sign = 1;
  int val = 0;
  if (len == consume)
    return false;
  while (1) {
    if (len == consume)
      return false;
    if (str[consume] == '-') {
      consume++; // Consume "-"
      sign *= -1;
    } else if (str[consume] == '+') {
      consume++; // Consume "+"
      sign *= 1;
    } else {
      break;
    }
  }
  if (len == consume)
    return false;
  if (str[consume] < '0' || '9' < str[consume])
    return false;
  val = int(str[consume] - '0');
  consume++; // Consume a digit.
  while (1) {
    if (len == consume || str[consume] < '0' || '9' < str[consume]) {
      int ret_val = sign * val;
      if (type == num_positive && ret_val <= 0) {
        return false;
      } else if (type == num_nonnegative && ret_val < 0) {
        return false;
      }
      *p_consume += consume;
      *p_val += sign * val;
      return true;
    }
    val = val * 10 + int(str[consume] - '0');
    consume++; // Consume a digit.
  }
  // Unreachable.
}

static inline int __kmp_abt_parse_mod_place(int val, int num_xstreams) {
  return ((val % num_xstreams) + num_xstreams) % num_xstreams;
}

typedef struct kmp_abt_parse_res_interval {
  parse_res_interval_t type;
  int res;
  int num_places;
  int stride;
} kmp_abt_parse_res_interval_t;

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_create
    (parse_res_interval_t type, unsigned int res, unsigned int num_places,
     int stride) {
  kmp_abt_parse_res_interval_t *p_parse_res_interval =
      (kmp_abt_parse_res_interval_t *)
      malloc(sizeof(kmp_abt_parse_res_interval_t));
  p_parse_res_interval->type = type;
  p_parse_res_interval->res = res;
  p_parse_res_interval->num_places = num_places;
  p_parse_res_interval->stride = stride;
  return p_parse_res_interval;
  }

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_create_rns
    (unsigned int res, unsigned int num_places, int stride) {
  return __kmp_abt_parse_res_interval_create(res_interval_rns, res, num_places,
                                             stride);
}

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_create_rn
    (unsigned int res, unsigned int num_places) {
  return __kmp_abt_parse_res_interval_create(res_interval_rn, res, num_places,
                                             0);
}

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_create_ir
    (unsigned int res) {
  return __kmp_abt_parse_res_interval_create(res_interval_ir, res, 0, 0);
}

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_create_r
    (unsigned int res) {
  return __kmp_abt_parse_res_interval_create(res_interval_r, res, 0, 0);
}

static kmp_abt_parse_res_interval_t *__kmp_abt_parse_res_interval_parse
    (const char *str, size_t len, size_t *p_consume) {
  bool invert = false;
  size_t consume = 0;
  if (len != consume && str[consume] == '!') {
    invert = true;
    consume++; // Consume "!"
  }
  int res = 0;
  if (!__kmp_abt_parse_num(&res, num_nonnegative, str + consume, len - consume,
                           &consume))
    return NULL;
  if (!invert && len != consume && str[consume] == ':') {
    consume++; // Consume ":"
    int num_places = 0;
    if (!__kmp_abt_parse_num(&num_places, num_positive, str + consume,
                             len - consume, &consume))
      return NULL;
    if (len != consume && str[consume] == ':') {
      consume++; // Consume ":"
      int stride = 0;
      if (!__kmp_abt_parse_num(&stride, num_any, str + consume, len - consume,
                               &consume))
        return NULL;
      *p_consume += consume;
      return __kmp_abt_parse_res_interval_create_rns(res, num_places, stride);
    } else {
      *p_consume += consume;
      return __kmp_abt_parse_res_interval_create_rn(res, num_places);
    }
  } else {
    *p_consume += consume;
    if (invert) {
      return __kmp_abt_parse_res_interval_create_ir(res);
    } else {
      return __kmp_abt_parse_res_interval_create_r(res);
    }
  }
}

static void __kmp_abt_parse_res_interval_free
    (kmp_abt_parse_res_interval_t *p_res_interval) {
  free(p_res_interval);
}

static kmp_abt_affinity_place_t *__kmp_abt_parse_res_interval_generate_place
    (const kmp_abt_parse_res_interval_t *p_res_interval, int num_xstreams) {
  kmp_abt_affinity_place_t *p_place = __kmp_abt_affinity_place_create();
  if (p_res_interval->type == res_interval_rns) {
    for (int i = 0; i < p_res_interval->num_places; i++)
      __kmp_abt_affinity_place_insert(p_place,
          __kmp_abt_parse_mod_place(p_res_interval->res + i
                                    * p_res_interval->stride, num_xstreams));
  } else if (p_res_interval->type == res_interval_rn) {
    for (int i = 0; i < p_res_interval->num_places; i++)
      __kmp_abt_affinity_place_insert(p_place,
          __kmp_abt_parse_mod_place(p_res_interval->res + i, num_xstreams));
  } else if (p_res_interval->type == res_interval_r) {
    __kmp_abt_affinity_place_insert(p_place,
        __kmp_abt_parse_mod_place(p_res_interval->res, num_xstreams));
  } else {
    for (int i = 0; i < num_xstreams; i++) {
      if (i != __kmp_abt_parse_mod_place(p_res_interval->res, num_xstreams))
        __kmp_abt_affinity_place_insert(p_place, i);
    }
  }
  return p_place;
}

static void __kmp_abt_parse_res_interval_print
    (const kmp_abt_parse_res_interval_t *p_res_interval) {
  if (p_res_interval->type == res_interval_rns) {
    printf("%d:%d:%d", (int)p_res_interval->res,
           (int)p_res_interval->num_places, (int)p_res_interval->stride);
  } else if (p_res_interval->type == res_interval_rn) {
    printf("%d:%d", (int)p_res_interval->res, (int)p_res_interval->num_places);
  } else if (p_res_interval->type == res_interval_r) {
    printf("%d", (int)p_res_interval->res);
  } else {
    printf("!%d", (int)p_res_interval->res);
  }
}

typedef struct kmp_abt_parse_res_list {
  size_t len_res_intervals;
  kmp_abt_parse_res_interval_t **p_res_intervals;
} kmp_abt_parse_res_list_t;

static kmp_abt_parse_res_list_t *__kmp_abt_parse_res_list_create() {
  kmp_abt_parse_res_list_t *p_res_list = (kmp_abt_parse_res_list_t *)
      malloc(sizeof(kmp_abt_parse_res_list_t));
  p_res_list->len_res_intervals = 0;
  p_res_list->p_res_intervals = NULL;
  return p_res_list;
}

static void __kmp_abt_parse_res_list_push_back
    (kmp_abt_parse_res_list_t *p_res_list,
     kmp_abt_parse_res_interval_t *p_res_interval) {
  size_t len_res_intervals = p_res_list->len_res_intervals;
  size_t new_len_res_intervals = len_res_intervals + 1;
  kmp_abt_parse_res_interval_t **p_new_res_intervals
      = (kmp_abt_parse_res_interval_t **)
        malloc(sizeof(kmp_abt_parse_res_interval_t *) * new_len_res_intervals);
  memcpy(p_new_res_intervals, p_res_list->p_res_intervals,
         sizeof(kmp_abt_parse_res_interval_t *) * len_res_intervals);
  free(p_res_list->p_res_intervals);
  p_new_res_intervals[len_res_intervals] = p_res_interval;
  p_res_list->len_res_intervals = new_len_res_intervals;
  p_res_list->p_res_intervals = p_new_res_intervals;
}

static void __kmp_abt_parse_res_list_free
    (kmp_abt_parse_res_list_t *p_res_list) {
  for (size_t i = 0; i < p_res_list->len_res_intervals; i++)
    __kmp_abt_parse_res_interval_free(p_res_list->p_res_intervals[i]);
  free(p_res_list->p_res_intervals);
  free(p_res_list);
}

static kmp_abt_parse_res_list_t *__kmp_abt_parse_res_list_parse
    (const char *str, size_t len, size_t *p_consume) {
  kmp_abt_parse_res_list_t *p_res_list = __kmp_abt_parse_res_list_create();
  size_t consume = 0;
  while(1) {
    kmp_abt_parse_res_interval_t *p_res_interval
        = __kmp_abt_parse_res_interval_parse(str + consume, len - consume,
                                             &consume);
    if (!p_res_interval) {
      __kmp_abt_parse_res_list_free(p_res_list);
      return NULL;
    }
    __kmp_abt_parse_res_list_push_back(p_res_list, p_res_interval);
    if (consume == len || str[consume] != ',') {
      *p_consume += consume;
      return p_res_list;
    }
    consume++; // Consume ","
  }
  // Unreachable.
}

static kmp_abt_affinity_place_t *__kmp_abt_parse_res_list_generate_place
    (const kmp_abt_parse_res_list_t *p_res_list, int num_xstreams) {
  kmp_abt_affinity_place_t *p_place = __kmp_abt_affinity_place_create();
  for (size_t i = 0; i < p_res_list->len_res_intervals; i++) {
    kmp_abt_parse_res_interval_t *p_res_interval
        = p_res_list->p_res_intervals[i];
    kmp_abt_affinity_place_t *p_ret_place
        = __kmp_abt_parse_res_interval_generate_place(p_res_interval,
                                                      num_xstreams);
    __kmp_abt_affinity_place_insert_place(p_place, p_ret_place);
    __kmp_abt_affinity_place_free(p_ret_place);
  }
  return p_place;
}

static void __kmp_abt_parse_res_list_print
    (const kmp_abt_parse_res_list_t *p_res_list) {
  int index = 0;
  for (size_t i = 0; i < p_res_list->len_res_intervals; i++) {
    if (index++ != 0)
      printf(",");
    __kmp_abt_parse_res_interval_print(p_res_list->p_res_intervals[i]);
  }
}

typedef struct kmp_abt_parse_pinterval {
  parse_pinterval_t type;
  kmp_abt_parse_res_list_t *p_res_list;
  int len;
  int stride;
} kmp_abt_parse_pinterval_t;

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_create
    (parse_pinterval_t type, kmp_abt_parse_res_list_t *p_res_list, int len,
     int stride) {
  kmp_abt_parse_pinterval_t *p_pinterval
      = (kmp_abt_parse_pinterval_t *)malloc(sizeof(kmp_abt_parse_pinterval_t));
  p_pinterval->type = type;
  p_pinterval->p_res_list = p_res_list;
  p_pinterval->len = len;
  p_pinterval->stride = stride;
  return p_pinterval;
}

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_create_pls
    (kmp_abt_parse_res_list_t *p_res_list, int len, int stride) {
  return __kmp_abt_parse_pinterval_create(pinterval_pls, p_res_list, len,
                                          stride);
}

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_create_pl
    (kmp_abt_parse_res_list_t *p_res_list, int len) {
  return __kmp_abt_parse_pinterval_create(pinterval_pl, p_res_list, len, 0);
}

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_create_ip
    (kmp_abt_parse_res_list_t *p_res_list) {
  return __kmp_abt_parse_pinterval_create(pinterval_ip, p_res_list, 0, 0);
}

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_create_p
    (kmp_abt_parse_res_list_t *p_res_list) {
  return __kmp_abt_parse_pinterval_create(pinterval_p, p_res_list, 0, 0);
}

static void __kmp_abt_parse_pinterval_free
    (kmp_abt_parse_pinterval_t *p_pinterval) {
  __kmp_abt_parse_res_list_free(p_pinterval->p_res_list);
  free(p_pinterval);
}

static kmp_abt_parse_pinterval_t *__kmp_abt_parse_pinterval_parse
    (const char *str, size_t len, size_t *p_consume) {
  bool invert = false;
  size_t consume = 0;
  if (len != consume && str[consume] == '!') {
    invert = true;
    consume++; // Consume "!"
  }
  if (len == consume || str[consume] != '{') {
    return NULL;
  } else {
    consume++; // Consume "{"
  }
  kmp_abt_parse_res_list_t *p_res_list
      = __kmp_abt_parse_res_list_parse(str + consume, len - consume, &consume);
  if (!p_res_list)
    return NULL;
  if (len == consume || str[consume] != '}') {
    __kmp_abt_parse_res_list_free(p_res_list);
    return NULL;
  } else {
    consume++; // Consume "{"
  }
  if (!invert && len != consume && str[consume] == ':') {
    consume++; // Consume ":"
    int len_val = 0;
    if (!__kmp_abt_parse_num(&len_val, num_positive, str + consume,
                             len - consume, &consume)) {
      __kmp_abt_parse_res_list_free(p_res_list);
      return NULL;
    }
    if (len != consume && str[consume] == ':') {
      consume++; // Consume ":"
      int stride = 0;
      if (!__kmp_abt_parse_num(&stride, num_any, str + consume, len - consume,
                               &consume)) {
        __kmp_abt_parse_res_list_free(p_res_list);
        return NULL;
      }
      *p_consume += consume;
      return __kmp_abt_parse_pinterval_create_pls(p_res_list, len_val, stride);
    } else {
      *p_consume += consume;
      return __kmp_abt_parse_pinterval_create_pl(p_res_list, len_val);
    }
  } else {
    *p_consume += consume;
    if (invert) {
      return __kmp_abt_parse_pinterval_create_ip(p_res_list);
    } else {
      return __kmp_abt_parse_pinterval_create_p(p_res_list);
    }
  }
}

static kmp_abt_affinity_places_t *__kmp_abt_parse_pinterval_generate_places
    (const kmp_abt_parse_pinterval_t *p_pinterval, int num_xstreams) {
  kmp_abt_affinity_place_t *p_place
      = __kmp_abt_parse_res_list_generate_place(p_pinterval->p_res_list,
                                                num_xstreams);
  kmp_abt_affinity_places_t *p_places = __kmp_abt_affinity_places_create();
  if (p_pinterval->type == pinterval_pls) {
    for (int i = 0, len = p_pinterval->len; i < len; i++) {
      kmp_abt_affinity_place_t *p_tmp_place = __kmp_abt_affinity_place_create();
      for (size_t j = 0; j != p_place->num_ranks; j++) {
        int val = p_place->ranks[j];
        __kmp_abt_affinity_place_insert(p_tmp_place,
            __kmp_abt_parse_mod_place(val + i * p_pinterval->stride,
                                      num_xstreams));
      }
      __kmp_abt_affinity_places_add(p_places, p_tmp_place);
    }
    __kmp_abt_affinity_place_free(p_place);
  } else if (p_pinterval->type == pinterval_pl) {
    for (int i = 0, len = p_pinterval->len; i < len; i++) {
      kmp_abt_affinity_place_t *p_tmp_place = __kmp_abt_affinity_place_create();
      for (size_t j = 0; j != p_place->num_ranks; j++) {
        int val = p_place->ranks[j];
        __kmp_abt_affinity_place_insert(p_tmp_place,
            __kmp_abt_parse_mod_place(val + i, num_xstreams));
      }
      __kmp_abt_affinity_places_add(p_places, p_tmp_place);
    }
    __kmp_abt_affinity_place_free(p_place);
  } else if (p_pinterval->type == pinterval_p) {
    __kmp_abt_affinity_places_add(p_places, p_place);
  } else {
    // Invert.
    kmp_abt_affinity_place_t *p_tmp_place = __kmp_abt_affinity_place_create();
    for (int i = 0; i < num_xstreams; i++) {
      if (!__kmp_abt_affinity_place_find(p_place, i))
        __kmp_abt_affinity_place_insert(p_tmp_place, i);
    }
    __kmp_abt_affinity_places_add(p_places, p_tmp_place);
    __kmp_abt_affinity_place_free(p_place);
  }
  return p_places;
}

static void __kmp_abt_parse_pinterval_print
    (const kmp_abt_parse_pinterval_t *p_pinterval) {
  if (p_pinterval->type == pinterval_pls) {
    printf("{");
    __kmp_abt_parse_res_list_print(p_pinterval->p_res_list);
    printf("}:%d:%d", (int)p_pinterval->len, (int)p_pinterval->stride);
  } else if (p_pinterval->type == pinterval_pl) {
    printf("{");
    __kmp_abt_parse_res_list_print(p_pinterval->p_res_list);
    printf("}:%d", (int)p_pinterval->len);
  } else if (p_pinterval->type == pinterval_p) {
    printf("{");
    __kmp_abt_parse_res_list_print(p_pinterval->p_res_list);
    printf("}");
  } else {
    printf("!{");
    __kmp_abt_parse_res_list_print(p_pinterval->p_res_list);
    printf("}");
  }
}

typedef struct kmp_abt_parse_aname {
  parse_word_t word;
  int num_places;
} kmp_abt_parse_aname_t;

static kmp_abt_parse_aname_t *__kmp_abt_parse_aname_create_p(parse_word_t word,
                                                             int num_places) {
  kmp_abt_parse_aname_t *p_aname
      = (kmp_abt_parse_aname_t *)malloc(sizeof(kmp_abt_parse_aname_t));
  p_aname->word = word;
  p_aname->num_places = num_places;
  return p_aname;
}

static kmp_abt_parse_aname_t *__kmp_abt_parse_aname_create(parse_word_t word) {
  return __kmp_abt_parse_aname_create_p(word, -1);
}

static void __kmp_abt_parse_aname_free(kmp_abt_parse_aname_t *p_aname) {
  free(p_aname);
}

static kmp_abt_parse_aname_t *__kmp_abt_parse_aname_parse(const char *str,
                                                          size_t len,
                                                          size_t *p_consume) {
  size_t consume = 0;
  parse_word_t word;
  if (len >= 7 && strncmp(str, "sockets", 7) == 0) {
    consume = 7;
    word = word_sockets;
  } else if (len >= 5 && strncmp(str, "cores", 5) == 0) {
    consume = 5;
    word = word_cores;
  } else if (len >= 7 && strncmp(str, "threads", 7) == 0) {
    consume = 7;
    word = word_threads;
  } else {
    return NULL;
  }
  if (len != consume && str[consume] == '(') {
    consume++; // Consume "("
    int num_places = 0;
    if (!__kmp_abt_parse_num(&num_places, num_positive, str + consume,
                             len - consume, &consume)) {
      return NULL;
    }
    if (len != consume && str[consume] == ')') {
      consume++; // Consume ")"
      *p_consume += consume;
      return __kmp_abt_parse_aname_create_p(word, num_places);
    }
  } else {
    *p_consume += consume;
    return __kmp_abt_parse_aname_create(word);
  }
  return NULL;
}

static kmp_abt_affinity_places_t *__kmp_abt_parse_aname_generate_places
    (const kmp_abt_parse_aname_t *p_aname, int num_xstreams) {
  kmp_abt_affinity_places_t *p_places = __kmp_abt_affinity_places_create();
  if (p_aname->word == word_sockets || p_aname->num_places == -1) {
    // Ignore.
    for (int i = 0; i < num_xstreams; i++)
      __kmp_abt_affinity_places_add(p_places,
                                    __kmp_abt_affinity_place_create_rank(i));
  } else {
    for (int i = 0; i < p_aname->num_places; i++) {
      int jstart = num_xstreams * i / p_aname->num_places;
      int jend = num_xstreams * (i + 1) / p_aname->num_places;
      kmp_abt_affinity_place_t *p_place = __kmp_abt_affinity_place_create();
      for (int j = jstart; j < jend; j++)
        __kmp_abt_affinity_place_insert(p_place, j);
      __kmp_abt_affinity_places_add(p_places, p_place);
    }
  }
  return p_places;
}

static void __kmp_abt_parse_aname_print(const kmp_abt_parse_aname_t *p_aname) {
  if (p_aname->word == word_sockets) {
    printf("sockets");
  } else if (p_aname->word == word_cores) {
    printf("cores");
  } else {
    printf("threads");
  }
  if (p_aname->num_places != -1) {
    printf("(%d)", (int)p_aname->num_places);
  }
}

typedef struct kmp_abt_parse_plist {
  size_t len_pintervals;
  kmp_abt_parse_pinterval_t **p_pintervals;
} kmp_abt_parse_plist_t;

static kmp_abt_parse_plist_t *__kmp_abt_parse_plist_create() {
  kmp_abt_parse_plist_t *p_plist
      = (kmp_abt_parse_plist_t *)malloc(sizeof(kmp_abt_parse_plist_t));
  p_plist->len_pintervals = 0;
  p_plist->p_pintervals = NULL;
  return p_plist;
}

static void __kmp_abt_parse_plist_free(kmp_abt_parse_plist_t *p_plist) {
  for (size_t i = 0; i < p_plist->len_pintervals; i++)
    __kmp_abt_parse_pinterval_free(p_plist->p_pintervals[i]);
  free(p_plist->p_pintervals);
  free(p_plist);
}

static void __kmp_abt_parse_plist_add(kmp_abt_parse_plist_t *p_plist,
                                      kmp_abt_parse_pinterval_t *p_pinterval) {
  size_t len_pintervals = p_plist->len_pintervals;
  size_t new_len_pintervals = len_pintervals + 1;
  kmp_abt_parse_pinterval_t **p_new_pintervals
      = (kmp_abt_parse_pinterval_t **)malloc(sizeof(kmp_abt_parse_pinterval_t *)
                                             * new_len_pintervals);
  memcpy(p_new_pintervals, p_plist->p_pintervals,
         sizeof(kmp_abt_parse_pinterval_t *) * len_pintervals);
  free(p_plist->p_pintervals);
  p_new_pintervals[len_pintervals] = p_pinterval;
  p_plist->len_pintervals = new_len_pintervals;
  p_plist->p_pintervals = p_new_pintervals;
}

static kmp_abt_parse_plist_t *__kmp_abt_parse_plist_parse(const char *str,
                                                          size_t len,
                                                          size_t *p_consume) {
  kmp_abt_parse_plist_t *p_plist = __kmp_abt_parse_plist_create();
  size_t consume = 0;
  while(1) {
    kmp_abt_parse_pinterval_t *p_pinterval
        = __kmp_abt_parse_pinterval_parse(str + consume, len - consume,
                                          &consume);
    if (!p_pinterval) {
      __kmp_abt_parse_plist_free(p_plist);
      return NULL;
    }
    __kmp_abt_parse_plist_add(p_plist, p_pinterval);
    if (consume == len || str[consume] != ',') {
      *p_consume += consume;
      return p_plist;
    }
    consume++; // Consume ","
  }
  // Unreachable.
}

static kmp_abt_affinity_places_t *__kmp_abt_parse_plist_generate_places
    (const kmp_abt_parse_plist_t *p_plist, int num_xstreams) {
  kmp_abt_affinity_places_t *p_places =  __kmp_abt_affinity_places_create();
  for (size_t i = 0; i < p_plist->len_pintervals; i++) {
    kmp_abt_affinity_places_t *p_ret_places
        = __kmp_abt_parse_pinterval_generate_places(p_plist->p_pintervals[i],
                                                    num_xstreams);
    for (size_t j = 0; j < p_ret_places->num_places; j++) {
      kmp_abt_affinity_place_t *p_place
        = __kmp_abt_affinity_place_create_place(p_ret_places->p_places[j]);
      __kmp_abt_affinity_places_add(p_places, p_place);
    }
    __kmp_abt_affinity_places_free(p_ret_places);
  }
  return p_places;
}

static void __kmp_abt_parse_plist_print(const kmp_abt_parse_plist_t *p_plist) {
  int index = 0;
  for (size_t i = 0; i < p_plist->len_pintervals; i++) {
    if (index++ != 0)
      printf(",");
    __kmp_abt_parse_pinterval_print(p_plist->p_pintervals[i]);
  }
}

typedef struct kmp_abt_parse_list {
  kmp_abt_parse_plist_t *p_plist;
  kmp_abt_parse_aname_t *p_aname;
} kmp_abt_parse_list_t;

static kmp_abt_parse_list_t *__kmp_abt_parse_list_create
    (kmp_abt_parse_plist_t *p_plist, kmp_abt_parse_aname_t *p_aname) {
  kmp_abt_parse_list_t *p_list
      = (kmp_abt_parse_list_t *)malloc(sizeof(kmp_abt_parse_list_t));
  p_list->p_plist = p_plist;
  p_list->p_aname = p_aname;
  return p_list;
}

static kmp_abt_parse_list_t *__kmp_abt_parse_list_create_plist
    (kmp_abt_parse_plist_t *p_plist) {
  return __kmp_abt_parse_list_create(p_plist, NULL);
}

static kmp_abt_parse_list_t *__kmp_abt_parse_list_create_aname
    (kmp_abt_parse_aname_t *p_aname) {
  return __kmp_abt_parse_list_create(NULL, p_aname);
}

static void __kmp_abt_parse_list_free(kmp_abt_parse_list_t *p_list) {
  if (p_list->p_plist)
    __kmp_abt_parse_plist_free(p_list->p_plist);
  if (p_list->p_aname)
    __kmp_abt_parse_aname_free(p_list->p_aname);
  free(p_list);
}

static kmp_abt_parse_list_t *__kmp_abt_parse_list_parse(const char *str,
                                                        size_t len,
                                                        size_t *p_consume) {
  kmp_abt_parse_plist_t *p_plist = __kmp_abt_parse_plist_parse(str, len,
                                                               p_consume);
  if (p_plist)
    return __kmp_abt_parse_list_create_plist(p_plist);
  kmp_abt_parse_aname_t *p_aname = __kmp_abt_parse_aname_parse(str, len,
                                                               p_consume);
  if (p_aname)
    return __kmp_abt_parse_list_create_aname(p_aname);
  return NULL;
}

static kmp_abt_affinity_places_t *__kmp_abt_parse_list_generate_places
    (const kmp_abt_parse_list_t *p_list, int num_xstreams) {
  if (p_list->p_plist) {
    return __kmp_abt_parse_plist_generate_places(p_list->p_plist, num_xstreams);
  } else {
    return __kmp_abt_parse_aname_generate_places(p_list->p_aname, num_xstreams);
  }
}

static void __kmp_abt_parse_list_print(const kmp_abt_parse_list_t *p_list) {
  if (p_list->p_plist) {
    __kmp_abt_parse_plist_print(p_list->p_plist);
  } else {
    __kmp_abt_parse_aname_print(p_list->p_aname);
  }
}

kmp_abt_affinity_places_t *__kmp_abt_parse_affinity(int num_xstreams,
                                                    const char *str, size_t len,
                                                    bool verbose) {
  // <list> |= <p-list> | <aname>
  // <p-list> |= <p-interval> | <p-list>,<p-interval>
  // <p-interval> |= <place>:<len>:<stride> | <place>:<len> | <place> | !<place>
  // <place> |= {<res-list>}
  // <res-list> |= <res-interval> | <res-list>,<res-interval>
  // <res-interval> |= <res>:<num-places>:<stride> | <res>:<num-places> | <res>
  //                   | !<res>
  // <aname> |= <word>(<num-places>) | <word>
  // <word> |= sockets | cores | threads
  //           | <implementation-defined abstract name>
  // <res> |= non-negative integer
  // <num-places> |= positive integer
  // <stride> |= integer
  // <len> |= positive integer

  size_t consume = 0;
  kmp_abt_parse_list_t *p_list = __kmp_abt_parse_list_parse(str, len, &consume);
  kmp_abt_affinity_places_t *p_places = NULL;
  if (!p_list) {
    if (verbose) {
      printf("parse failed:\n");
      printf("use default places.\n");
    }
    // Create a default one.
    p_places = __kmp_abt_affinity_places_create();
    for (int i = 0; i < num_xstreams; i++) {
      kmp_abt_affinity_place_t *p_place
          = __kmp_abt_affinity_place_create_rank(i);
      __kmp_abt_affinity_places_add(p_places, p_place);
    }
  } else {
    if (verbose) {
      printf("parse succeeded:\n");
      printf("  %s\n->", str);
      __kmp_abt_parse_list_print(p_list);
      printf("\n");
    }
    p_places = __kmp_abt_parse_list_generate_places(p_list, num_xstreams);
    __kmp_abt_parse_list_free(p_list);
  }
  if (verbose) {
    for (size_t i = 0; i < p_places->num_places; i++) {
      if (i != 0)
        printf(",");
      printf("[%d]:{", (int)i);
      bool is_first = true;
      kmp_abt_affinity_place_t *p_place = p_places->p_places[i];
      for (size_t j = 0; j < p_place->num_ranks; j++) {
        if (!is_first)
          printf(",");
        is_first = false;
        printf("%d", p_place->ranks[j]);
      }
      printf("}");
    }
    printf("\n");
  }
  return p_places;
}
