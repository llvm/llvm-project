/* Prototypes for the functions in the DSOs.  */
extern int calllocal1 (void);
extern int (*getlocal1 (void)) (void);
extern int callinmod1 (void);
extern int (*getinmod1 (void)) (void);
extern int callitcpt1 (void);
extern int (*getitcpt1 (void)) (void);
extern const char **getvarlocal1 (void);
extern const char **getvarinmod1 (void);
extern const char **getvaritcpt1 (void);
extern int calllocal2 (void);
extern int (*getlocal2 (void)) (void);
extern int callinmod2 (void);
extern int (*getinmod2 (void)) (void);
extern int callitcpt2 (void);
extern int (*getitcpt2 (void)) (void);
extern const char **getvarlocal2 (void);
extern const char **getvarinmod2 (void);
extern const char **getvaritcpt2 (void);
extern int callitcpt3 (void);
extern int (*getitcpt3 (void)) (void);
extern const char **getvaritcpt3 (void);

extern int protinmod (void);
extern int protitcpt (void);
extern int protlocal (void);

