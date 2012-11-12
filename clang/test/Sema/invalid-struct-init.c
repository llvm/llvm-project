// RUN: %clang_cc1 %s -verify -fsyntax-only

typedef struct _zend_module_entry zend_module_entry;
struct _zend_module_entry {
  _efree((p)); // expected-error{{type name requires a specifier or qualifier}} \
                  expected-warning {{type specifier missing, defaults to 'int'}}
  
};
typedef struct _zend_function_entry { } zend_function_entry;
typedef struct _zend_pcre_globals { } zend_pcre_globals;
zend_pcre_globals pcre_globals;

static void zm_globals_ctor_pcre(zend_pcre_globals *pcre_globals ) { }
static void zm_globals_dtor_pcre(zend_pcre_globals *pcre_globals ) { }
static void zm_info_pcre(zend_module_entry *zend_module ) { }
static int zm_startup_pcre(int type, int module_number ) { }

static int zm_shutdown_pcre(int type, int module_number ) {
  zend_function_entry pcre_functions[] = {{ }; // expected-error{{expected '}'}} expected-note {{to match this '{'}}
  zend_module_entry pcre_module_entry = {
    sizeof(zend_module_entry), 20071006, 0, 0, ((void *)0), ((void *)0),    
    "pcre",  pcre_functions,  zm_startup_pcre,  zm_shutdown_pcre,  ((void *)0),  
    ((void *)0),  zm_info_pcre,  ((void *)0),  sizeof(zend_pcre_globals), &pcre_globals,  
    ((void (*)(void* ))(zm_globals_ctor_pcre)),  ((void (*)(void* ))(zm_globals_dtor_pcre)),  
    ((void *)0),  0, 0, ((void *)0), 0 
  };
}
