// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-c++2d-extensions -Wno-c++23-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wno-c++2d-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++26 -fsyntax-only -Wno-c++2d-extensions --embed-dir=%S/Inputs %s
// RUN: %clang_cc1 -std=c++2d -fsyntax-only --embed-dir=%S/Inputs %s
// expected-no-diagnostics

#depend "resources/*"

namespace std {
  enum class byte : unsigned char {};
  typedef decltype(sizeof(0)) size_t;
}

// 79 x 40
constexpr const unsigned char v_correct[] = 
// clang-format off
u8"                                         @#=+@@                 @@%%@@         "
u8"                     R                  @======%@            @@%%%%%%%@        "
u8"                                         %=*%%==============*##%%%%%%%@        "
u8"                                          %=%%%*============*###%%%%%@         "
u8"                     A                     @=%+=========+****##%%%%%%          "
u8"                                             *==+----===+-----+#%%%%@          "
u8"                                            @==-----====--%%*---=%%%@          "
u8"                     C                      +=*-#%@@%-===%%@@%%%--%%%@         "
u8"                                            #*-%%%%%*-===-%@%%%#%#**%@         "
u8"                                            @+**%%%=--**--+*%@@@@#**%%         "
u8"                     C                       *+=+-=--@%%%#-=****++***%@        "
u8"                                             @***+==--===---#+*******%%@       "
u8"                                             @*==**%%%#%%%%**+*******%%@       "
u8"                                             @+===+**%%#**************%%@      "
u8"                                            %========*****************%%%@     "
u8"                                          @======*********************#%%@     "
u8"                                         *========********************##%%@    "
u8"                                        +=========*******************###%%@    "
u8"                                     @===========+******************#####%@    "
u8"                                   #=============+****************#######%@    "
u8"                                 @+==============****************########%%    "
u8"                                %================+**************##########%@   "
u8"                               @@=================*************###########%%@  "
u8"                                @=================*************###########%%@  "
u8"                                +==================************############%@  "
u8"                              @@@===================+**********############%%  "
u8"                                *================+===**********###########%@@  "
u8"                               @+================*+==+*********##########%%@@  "
u8"                                @==============%%*+===********###########%%@   "
u8"                                @+=============%%**+==+*******#%#%#######%@    "
u8"                                 %=============%%%#**==*******#%%%#######@     "
u8"                 @@@  @ @@@      @=============*#%###*+*******#%%%%%#####@     "
u8"      @@@#-----##%%%%-----%%%%----++============#######******##%%%%%%####@     "
u8"   @-%%%%%%-----*%%%%%----*%%%-===#*======+#====@########*=+*###%%%%@@###@     "
u8" @%=--=%%%%%-----%%%%%+----%%%#===*%====%%%%%%%#@@#########=####%@@@ @###@     "
u8"@%%%%--%%%%%-----%%%%------%%%%==--%%====#%%@@@@@@@%@%#####@%####@@@@#####@    "
u8"@@%%#---%%%*====@@@*+*####@@@%=--=%%%%+=+###@@@%%@@@@@@@@@@@######@@@######@   "
u8"                                     @@@@@@                 @#######@@########@"
u8"                                                            @#########@@#@@##@ "
u8"                                                             @########@   \xF0\x9F\xA6\x9D";
// clang-format on
constexpr const std::size_t v_correct_size = sizeof(v_correct) - 1;
constexpr const unsigned char* v_correct_end = v_correct + v_correct_size;

template <typename I0, typename S0, typename I1, typename S1>
constexpr bool byte_equal (I0 first0, S0 last0, I1 first1, S1 last1) {
  for (;first0 != last0 || first1 != last1; ++first0, ++first1) {
    if ((unsigned char)*first0 != (unsigned char)*first1) {
      return false;
    }
  }
  return first0 == last0 && first1 == last1;
}

inline constexpr std::size_t size_max = (std::size_t)0xFFFFFFFFFFFFFFFFull;
inline constexpr std::size_t sentinel_size = 44203zu;
template <typename T>
inline constexpr T sentinel_object = {};
template <typename T>
inline constexpr const T* sentinel_address = &sentinel_object<T>;

template <typename T>
struct inspect {
  const T* ptr = sentinel_address<T>;
  std::size_t size = sentinel_size;
  int status = -1;
};

enum : int {
  not_found,
  found,
  no_depend,
  found_empty
};

template <typename T, typename C, std::size_t N>
consteval inspect<T> non_local_lookup_core (const C (&name)[N],
                                            std::size_t offset = 0,
                                            std::size_t limit = size_max) {
  inspect<T> result;
  if (limit == size_max)
    result.ptr = __builtin_std_embed(0b000, result.status, result.size,
                                     result.ptr, N-1, name, offset);
  else
    result.ptr = __builtin_std_embed(0b000, result.status, result.size,
                                     result.ptr, N-1, name, offset, limit);
  return result;
}

consteval bool art () {
  constexpr auto v0 = non_local_lookup_core<unsigned char>("resources/art.bin");
  static_assert(v0.status == found);
  static_assert(byte_equal(v0.ptr, v0.ptr + v0.size, &v_correct[0], v_correct_end));
  static_assert(v0.size == v_correct_size);
  constexpr auto v1 = non_local_lookup_core<unsigned char>(L"resources/art.bin");
  static_assert(v1.status == found);
  static_assert(byte_equal(v1.ptr, v1.ptr + v1.size, &v_correct[0], v_correct_end));
  static_assert(v1.size == v_correct_size);
  constexpr auto v2 = non_local_lookup_core<unsigned char>(u8"resources/art.bin");
  static_assert(v2.status == found);
  static_assert(byte_equal(v2.ptr, v2.ptr + v2.size, &v_correct[0], v_correct_end));
  static_assert(v2.size == v_correct_size);

  constexpr auto v3 = non_local_lookup_core<char>("resources/art.bin");
  static_assert(v3.status == found);
  static_assert(byte_equal(v3.ptr, v3.ptr + v3.size, &v_correct[0], v_correct_end));
  static_assert(v3.size == v_correct_size);
  constexpr auto v4 = non_local_lookup_core<char>(L"resources/art.bin");
  static_assert(v4.status == found);
  static_assert(byte_equal(v4.ptr, v4.ptr + v4.size, &v_correct[0], v_correct_end));
  static_assert(v4.size == v_correct_size);
  constexpr auto v5 = non_local_lookup_core<char>(u8"resources/art.bin");
  static_assert(v5.status == found);
  static_assert(byte_equal(v5.ptr, v5.ptr + v5.size, &v_correct[0], v_correct_end));
  static_assert(v5.size == v_correct_size);

  constexpr auto v6 = non_local_lookup_core<std::byte>("resources/art.bin");
  static_assert(v6.status == found);
  static_assert(byte_equal(v6.ptr, v6.ptr + v6.size, &v_correct[0], v_correct_end));
  static_assert(v6.size == v_correct_size);
  constexpr auto v7 = non_local_lookup_core<std::byte>(L"resources/art.bin");
  static_assert(v7.status == found);
  static_assert(byte_equal(v7.ptr, v7.ptr + v7.size, &v_correct[0], v_correct_end));
  static_assert(v7.size == v_correct_size);
  constexpr auto v8 = non_local_lookup_core<std::byte>(u8"resources/art.bin");
  static_assert(v8.status == found);
  static_assert(byte_equal(v8.ptr, v8.ptr + v8.size, &v_correct[0], v_correct_end));
  static_assert(v8.size == v_correct_size);

  return true;
}

consteval bool cannot_depend_on () {
  // test that #depend "resources/*" pattern deos not allow something
  // that is more "deeply" nested
  constexpr auto v0 = non_local_lookup_core<unsigned char>("resources/a/b/empty");
  static_assert(v0.status == no_depend);
  static_assert(v0.ptr == nullptr);
  static_assert(v0.size == sentinel_size);
  
  return true;
}

consteval bool offset () {
  constexpr std::size_t R_offset = 100;
  constexpr std::size_t A_offset = 237;
  constexpr auto v0 = non_local_lookup_core<unsigned char>("resources/art.bin", R_offset);
  static_assert(v0.status == found);
  static_assert(v0.ptr[0] == (unsigned char)'R');
  static_assert(v0.ptr[A_offset] == (unsigned char)'A');
  static_assert(v0.size == v_correct_size - R_offset);

  constexpr auto v1 = non_local_lookup_core<unsigned char>("resources/art.bin", size_max);
  static_assert(v1.status == found_empty);
  static_assert(v1.ptr == nullptr);
  static_assert(v1.size == 0);

  return true;
}

consteval bool limit () {
  constexpr std::size_t line_offset = 553;
  constexpr std::size_t line_limit = 79;
  constexpr std::size_t C_offset = 21;
  constexpr auto v0 = non_local_lookup_core<unsigned char>("resources/art.bin", line_offset, line_limit);
  static_assert(v0.status == found);
  static_assert(v0.ptr[C_offset] == (unsigned char)'C');
  static_assert(v0.size == line_limit);
  static_assert(byte_equal(v0.ptr, v0.ptr + v0.size, v_correct + line_offset, v_correct + line_offset + line_limit));

  constexpr auto v1 = non_local_lookup_core<unsigned char>("resources/art.bin", 0, 0);
  static_assert(v1.status == found_empty);
  static_assert(v1.ptr == nullptr);
  static_assert(v1.size == 0);

  return true;
}

static_assert(art());
static_assert(cannot_depend_on());
static_assert(offset());
static_assert(limit());
