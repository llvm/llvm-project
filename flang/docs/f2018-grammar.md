# Fortran 2018 Grammar

Grammar used by Flang to parse Fortran 2018.

```
R0001 digit -> 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
R0002 letter ->
        A | B | C | D | E | F | G | H | I | J | K | L | M |
        N | O | P | Q | R | S | T | U | V | W | X | Y | Z
R0003 rep-char

R401 xzy-list -> xzy [, xzy]...
R402 xzy-name -> name
R403 scalar-xyz -> xyz

R501 program -> program-unit [program-unit]...
R502 program-unit ->
       main-program | external-subprogram | module | submodule | block-data
R503 external-subprogram -> function-subprogram | subroutine-subprogram
R504 specification-part ->
       [use-stmt]... [import-stmt]... [implicit-part]
       [declaration-construct]...
R505 implicit-part -> [implicit-part-stmt]... implicit-stmt
R506 implicit-part-stmt ->
       implicit-stmt | parameter-stmt | format-stmt | entry-stmt
R507 declaration-construct ->
       specification-construct | data-stmt | format-stmt | entry-stmt |
       stmt-function-stmt
R508 specification-construct ->
       derived-type-def | enum-def | generic-stmt | interface-block |
       parameter-stmt | procedure-declaration-stmt |
       other-specification-stmt | type-declaration-stmt
R509 execution-part -> executable-construct [execution-part-construct]...
R510 execution-part-construct ->
       executable-construct | format-stmt | entry-stmt | data-stmt
R511 internal-subprogram-part -> contains-stmt [internal-subprogram]...
R512 internal-subprogram -> function-subprogram | subroutine-subprogram
R513 other-specification-stmt ->
       access-stmt | allocatable-stmt | asynchronous-stmt | bind-stmt |
       codimension-stmt | contiguous-stmt | dimension-stmt | external-stmt |
       intent-stmt | intrinsic-stmt | namelist-stmt | optional-stmt |
       pointer-stmt | protected-stmt | save-stmt | target-stmt |
       volatile-stmt | value-stmt | common-stmt | equivalence-stmt
R514 executable-construct ->
       action-stmt | associate-construct | block-construct | case-construct |
       change-team-construct | critical-construct | do-construct |
       if-construct | select-rank-construct | select-type-construct |
       where-construct | forall-construct
R515 action-stmt ->
       allocate-stmt | assignment-stmt | backspace-stmt | call-stmt |
       close-stmt | continue-stmt | cycle-stmt | deallocate-stmt |
       endfile-stmt | error-stop-stmt | event-post-stmt | event-wait-stmt |
       exit-stmt | fail-image-stmt | flush-stmt | form-team-stmt |
       goto-stmt | if-stmt | inquire-stmt | lock-stmt | nullify-stmt |
       open-stmt | pointer-assignment-stmt | print-stmt | read-stmt |
       return-stmt | rewind-stmt | stop-stmt | sync-all-stmt |
       sync-images-stmt | sync-memory-stmt | sync-team-stmt | unlock-stmt |
       wait-stmt | where-stmt | write-stmt | computed-goto-stmt | forall-stmt
R516 keyword -> name

R601 alphanumeric-character -> letter | digit | underscore  @ | $
R602 underscore -> _
R603 name -> letter [alphanumeric-character]...
R604 constant -> literal-constant | named-constant
R605 literal-constant ->
       int-literal-constant | real-literal-constant |
       complex-literal-constant | logical-literal-constant |
       char-literal-constant | boz-literal-constant
R606 named-constant -> name
R607 int-constant -> constant
R608 intrinsic-operator ->
       power-op | mult-op | add-op | concat-op | rel-op |
       not-op | and-op | or-op | equiv-op
R609 defined-operator ->
       defined-unary-op | defined-binary-op | extended-intrinsic-op
R610 extended-intrinsic-op -> intrinsic-operator
R611 label -> digit [digit]...
R620 delimiter -> ( | ) | / | [ | ] | (/ | /)

R701 type-param-value -> scalar-int-expr | * | :
R702 type-spec -> intrinsic-type-spec | derived-type-spec
R703 declaration-type-spec ->
       intrinsic-type-spec | TYPE ( intrinsic-type-spec ) |
       TYPE ( derived-type-spec ) | CLASS ( derived-type-spec ) |
       CLASS ( * ) | TYPE ( * )
R704 intrinsic-type-spec ->
       integer-type-spec | REAL [kind-selector] | DOUBLE PRECISION |
       COMPLEX [kind-selector] | CHARACTER [char-selector] |
       LOGICAL [kind-selector] @ DOUBLE COMPLEX
R705 integer-type-spec -> INTEGER [kind-selector]
R706 kind-selector ->
       ( [KIND =] scalar-int-constant-expr )  @ * scalar-int-constant-expr
R707 signed-int-literal-constant -> [sign] int-literal-constant
R708 int-literal-constant -> digit-string [_ kind-param]
R709 kind-param -> digit-string | scalar-int-constant-name
R710 signed-digit-string -> [sign] digit-string
R711 digit-string -> digit [digit]...
R712 sign -> + | -
R713 signed-real-literal-constant -> [sign] real-literal-constant
R714 real-literal-constant ->
       significand [exponent-letter exponent] [_ kind-param] |
       digit-string exponent-letter exponent [_ kind-param]
R715 significand -> digit-string . [digit-string] | . digit-string
R716 exponent-letter -> E | D  @ | Q
R717 exponent -> signed-digit-string
R718 complex-literal-constant -> ( real-part , imag-part )
R719 real-part ->
       signed-int-literal-constant | signed-real-literal-constant |
       named-constant
R720 imag-part ->
       signed-int-literal-constant | signed-real-literal-constant |
       named-constant
R721 char-selector ->
       length-selector |
       ( LEN = type-param-value , KIND = scalar-int-constant-expr ) |
       ( type-param-value , [KIND =] scalar-int-constant-expr ) |
       ( KIND = scalar-int-constant-expr [, LEN = type-param-value] )
R722 length-selector -> ( [LEN =] type-param-value ) | * char-length [,]
R723 char-length -> ( type-param-value ) | digit-string
R724 char-literal-constant ->
       [kind-param _] ' [rep-char]... ' | [kind-param _] " [rep-char]... "
R725 logical-literal-constant ->
       .TRUE. [_ kind-param] | .FALSE. [_ kind-param] @ | .T. | .F.
R726 derived-type-def ->
       derived-type-stmt [type-param-def-stmt]... [private-or-sequence]...
       [component-part] [type-bound-procedure-part] end-type-stmt
R727 derived-type-stmt ->
       TYPE [[, type-attr-spec-list] ::] type-name [( type-param-name-list )]
R728 type-attr-spec ->
       ABSTRACT | access-spec | BIND(C) | EXTENDS ( parent-type-name )
R729 private-or-sequence -> private-components-stmt | sequence-stmt
R730 end-type-stmt -> END TYPE [type-name]
R731 sequence-stmt -> SEQUENCE
R732 type-param-def-stmt ->
       integer-type-spec , type-param-attr-spec :: type-param-decl-list
R733 type-param-decl -> type-param-name [= scalar-int-constant-expr]
R734 type-param-attr-spec -> KIND | LEN
R735 component-part -> [component-def-stmt]...
R736 component-def-stmt -> data-component-def-stmt | proc-component-def-stmt
R737 data-component-def-stmt ->
       declaration-type-spec [[, component-attr-spec-list] ::]
       component-decl-list
R738 component-attr-spec ->
       access-spec | ALLOCATABLE |
       CODIMENSION lbracket coarray-spec rbracket | CONTIGUOUS |
       DIMENSION ( component-array-spec ) | POINTER
R739 component-decl ->
       component-name [( component-array-spec )]
       [lbracket coarray-spec rbracket] [* char-length]
       [component-initialization]
R740 component-array-spec ->
       explicit-shape-spec-list | deferred-shape-spec-list
R741 proc-component-def-stmt ->
       PROCEDURE ( [proc-interface] ) , proc-component-attr-spec-list ::
        proc-decl-list
R742 proc-component-attr-spec ->
       access-spec | NOPASS | PASS [(arg-name)] | POINTER
R743 component-initialization ->
       = constant-expr | => null-init | => initial-data-target
R744 initial-data-target -> designator
R745 private-components-stmt -> PRIVATE
R746 type-bound-procedure-part ->
       contains-stmt [binding-private-stmt] [type-bound-proc-binding]...
R747 binding-private-stmt -> PRIVATE
R748 type-bound-proc-binding ->
       type-bound-procedure-stmt | type-bound-generic-stmt |
       final-procedure-stmt
R749 type-bound-procedure-stmt ->
       PROCEDURE [[, bind-attr-list] ::] type-bound-proc-decl-list |
       PROCEDURE ( interface-name ) , bind-attr-list :: binding-name-list
R750 type-bound-proc-decl -> binding-name [=> procedure-name]
R751 type-bound-generic-stmt ->
       GENERIC [, access-spec] :: generic-spec => binding-name-list
R752 bind-attr ->
       access-spec | DEFERRED | NON_OVERRIDABLE | NOPASS | PASS [(arg-name)]
R753 final-procedure-stmt -> FINAL [::] final-subroutine-name-list
R754 derived-type-spec -> type-name [(type-param-spec-list)]
R755 type-param-spec -> [keyword =] type-param-value
R756 structure-constructor -> derived-type-spec ( [component-spec-list] )
R757 component-spec -> [keyword =] component-data-source
R758 component-data-source -> expr | data-target | proc-target
R759 enum-def ->
       enum-def-stmt enumerator-def-stmt [enumerator-def-stmt]... end-enum-stmt
R760 enum-def-stmt -> ENUM, BIND(C)
R761 enumerator-def-stmt -> ENUMERATOR [::] enumerator-list
R762 enumerator -> named-constant [= scalar-int-constant-expr]
R763 end-enum-stmt -> END ENUM
R764 boz-literal-constant -> binary-constant | octal-constant | hex-constant
R765 binary-constant -> B ' digit [digit]... ' | B " digit [digit]... "
R766 octal-constant -> O ' digit [digit]... ' | O " digit [digit]... "
R767 hex-constant ->
       Z ' hex-digit [hex-digit]... ' | Z " hex-digit [hex-digit]... "
R768 hex-digit -> digit | A | B | C | D | E | F
R769 array-constructor -> (/ ac-spec /) | lbracket ac-spec rbracket
R770 ac-spec -> type-spec :: | [type-spec ::] ac-value-list
R771 lbracket -> [
R772 rbracket -> ]
R773 ac-value -> expr | ac-implied-do
R774 ac-implied-do -> ( ac-value-list , ac-implied-do-control )
R775 ac-implied-do-control ->
       [integer-type-spec ::] ac-do-variable = scalar-int-expr ,
       scalar-int-expr [, scalar-int-expr]
R776 ac-do-variable -> do-variable

R801 type-declaration-stmt ->
       declaration-type-spec [[, attr-spec]... ::] entity-decl-list
R802 attr-spec ->
       access-spec | ALLOCATABLE | ASYNCHRONOUS |
       CODIMENSION lbracket coarray-spec rbracket | CONTIGUOUS |
       DIMENSION ( array-spec ) | EXTERNAL | INTENT ( intent-spec ) |
       INTRINSIC | language-binding-spec | OPTIONAL | PARAMETER |
       POINTER | PROTECTED | SAVE | TARGET | VALUE | VOLATILE
R803 entity-decl ->
       object-name [( array-spec )] [lbracket coarray-spec rbracket]
         [* char-length] [initialization] |
       function-name [* char-length]
R804 object-name -> name
R805 initialization -> = constant-expr | => null-init | => initial-data-target
R806 null-init -> function-reference
R807 access-spec -> PUBLIC | PRIVATE
R808 language-binding-spec ->
       BIND ( C [, NAME = scalar-default-char-constant-expr] )
R809 coarray-spec -> deferred-coshape-spec-list | explicit-coshape-spec
R810 deferred-coshape-spec -> :
R811 explicit-coshape-spec ->
       [[lower-cobound :] upper-cobound ,]... [lower-cobound :] *
R812 lower-cobound -> specification-expr
R813 upper-cobound -> specification-expr
R814 dimension-spec -> DIMENSION ( array-spec )
R815 array-spec ->
       explicit-shape-spec-list | assumed-shape-spec-list |
       deferred-shape-spec-list | assumed-size-spec | implied-shape-spec |
       implied-shape-or-assumed-size-spec | assumed-rank-spec
R816 explicit-shape-spec -> [lower-bound :] upper-bound
R817 lower-bound -> specification-expr
R818 upper-bound -> specification-expr
R819 assumed-shape-spec -> [lower-bound] :
R820 deferred-shape-spec -> :
R821 assumed-implied-spec -> [lower-bound :] *
R822 assumed-size-spec -> explicit-shape-spec-list , assumed-implied-spec
R823 implied-shape-or-assumed-size-spec -> assumed-implied-spec
R824 implied-shape-spec -> assumed-implied-spec , assumed-implied-spec-list
R825 assumed-rank-spec -> ..
R826 intent-spec -> IN | OUT | INOUT
R827 access-stmt -> access-spec [[::] access-id-list]
R828 access-id -> access-name | generic-spec
R829 allocatable-stmt -> ALLOCATABLE [::] allocatable-decl-list
R830 allocatable-decl ->
       object-name [( array-spec )] [lbracket coarray-spec rbracket]
R831 asynchronous-stmt -> ASYNCHRONOUS [::] object-name-list
R832 bind-stmt -> language-binding-spec [::] bind-entity-list
R833 bind-entity -> entity-name | / common-block-name /
R834 codimension-stmt -> CODIMENSION [::] codimension-decl-list
R835 codimension-decl -> coarray-name lbracket coarray-spec rbracket
R836 contiguous-stmt -> CONTIGUOUS [::] object-name-list
R837 data-stmt -> DATA data-stmt-set [[,] data-stmt-set]...
R838 data-stmt-set -> data-stmt-object-list / data-stmt-value-list /
R839 data-stmt-object -> variable | data-implied-do
R840 data-implied-do ->
       ( data-i-do-object-list , [integer-type-spec ::]
       data-i-do-variable = scalar-int-constant-expr ,
       scalar-int-constant-expr [, scalar-int-constant-expr] )
R841 data-i-do-object ->
       array-element | scalar-structure-component | data-implied-do
R842 data-i-do-variable -> do-variable
R843 data-stmt-value -> [data-stmt-repeat *] data-stmt-constant
R844 data-stmt-repeat -> scalar-int-constant | scalar-int-constant-subobject
R845 data-stmt-constant ->
       scalar-constant | scalar-constant-subobject |
       signed-int-literal-constant | signed-real-literal-constant |
       null-init | initial-data-target | structure-constructor
R846 int-constant-subobject -> constant-subobject
R847 constant-subobject -> designator
R848 dimension-stmt ->
       DIMENSION [::] array-name ( array-spec )
       [, array-name ( array-spec )]...
R849 intent-stmt -> INTENT ( intent-spec ) [::] dummy-arg-name-list
R850 optional-stmt -> OPTIONAL [::] dummy-arg-name-list
R851 parameter-stmt -> PARAMETER ( named-constant-def-list )
R852 named-constant-def -> named-constant = constant-expr
R853 pointer-stmt -> POINTER [::] pointer-decl-list
R854 pointer-decl ->
       object-name [( deferred-shape-spec-list )] | proc-entity-name
R855 protected-stmt -> PROTECTED [::] entity-name-list
R856 save-stmt -> SAVE [[::] saved-entity-list]
R857 saved-entity -> object-name | proc-pointer-name | / common-block-name /
R858 proc-pointer-name -> name
R859 target-stmt -> TARGET [::] target-decl-list
R860 target-decl ->
       object-name [( array-spec )] [lbracket coarray-spec rbracket]
R861 value-stmt -> VALUE [::] dummy-arg-name-list
R862 volatile-stmt -> VOLATILE [::] object-name-list
R863 implicit-stmt ->
       IMPLICIT implicit-spec-list |
       IMPLICIT NONE [( [implicit-name-spec-list] )]
R864 implicit-spec -> declaration-type-spec ( letter-spec-list )
R865 letter-spec -> letter [- letter]
R866 implicit-name-spec -> EXTERNAL | TYPE
R867 import-stmt ->
       IMPORT [[::] import-name-list] | IMPORT , ONLY : import-name-list |
       IMPORT , NONE | IMPORT , ALL
R868 namelist-stmt ->
       NAMELIST / namelist-group-name / namelist-group-object-list
       [[,] / namelist-group-name / namelist-group-object-list]...
R869 namelist-group-object -> variable-name
R870 equivalence-stmt -> EQUIVALENCE equivalence-set-list
R871 equivalence-set -> ( equivalence-object , equivalence-object-list )
R872 equivalence-object -> variable-name | array-element | substring
R873 common-stmt ->
       COMMON [/ [common-block-name] /] common-block-object-list
       [[,] / [common-block-name] / common-block-object-list]...
R874 common-block-object -> variable-name [( array-spec )]

R901 designator ->
       object-name | array-element | array-section |
       coindexed-named-object | complex-part-designator |
       structure-component | substring
R902 variable -> designator | function-reference
R903 variable-name -> name
R904 logical-variable -> variable
R905 char-variable -> variable
R906 default-char-variable -> variable
R907 int-variable -> variable
R908 substring -> parent-string ( substring-range )
R909 parent-string ->
       scalar-variable-name | array-element | coindexed-named-object |
       scalar-structure-component | scalar-char-literal-constant |
       scalar-named-constant
R910 substring-range -> [scalar-int-expr] : [scalar-int-expr]
R911 data-ref -> part-ref [% part-ref]...
R912 part-ref -> part-name [( section-subscript-list )] [image-selector]
R913 structure-component -> data-ref
R914 coindexed-named-object -> data-ref
R915 complex-part-designator -> designator % RE | designator % IM
R916 type-param-inquiry -> designator % type-param-name
R917 array-element -> data-ref
R918 array-section ->
       data-ref [( substring-range )] | complex-part-designator
R919 subscript -> scalar-int-expr
R920 section-subscript -> subscript | subscript-triplet | vector-subscript
R921 subscript-triplet -> [subscript] : [subscript] [: stride]
R922 stride -> scalar-int-expr
R923 vector-subscript -> int-expr
R924 image-selector ->
       lbracket cosubscript-list [, image-selector-spec-list] rbracket
R925 cosubscript -> scalar-int-expr
R926 image-selector-spec ->
       STAT = stat-variable | TEAM = team-value |
       TEAM_NUMBER = scalar-int-expr
R927 allocate-stmt ->
       ALLOCATE ( [type-spec ::] allocation-list [, alloc-opt-list] )
R928 alloc-opt ->
       ERRMSG = errmsg-variable | MOLD = source-expr |
       SOURCE = source-expr | STAT = stat-variable
R929 stat-variable -> scalar-int-variable
R930 errmsg-variable -> scalar-default-char-variable
R931 source-expr -> expr
R932 allocation ->
       allocate-object [( allocate-shape-spec-list )]
       [lbracket allocate-coarray-spec rbracket]
R933 allocate-object -> variable-name | structure-component
R934 allocate-shape-spec -> [lower-bound-expr :] upper-bound-expr
R935 lower-bound-expr -> scalar-int-expr
R936 upper-bound-expr -> scalar-int-expr
R937 allocate-coarray-spec ->
       [allocate-coshape-spec-list ,] [lower-bound-expr :] *
R938 allocate-coshape-spec -> [lower-bound-expr :] upper-bound-expr
R939 nullify-stmt -> NULLIFY ( pointer-object-list )
R940 pointer-object -> variable-name | structure-component | proc-pointer-name
R941 deallocate-stmt ->
       DEALLOCATE ( allocate-object-list [, dealloc-opt-list] )
R942 dealloc-opt -> STAT = stat-variable | ERRMSG = errmsg-variable

R1001 primary ->
        literal-constant | designator | array-constructor |
        structure-constructor | function-reference | type-param-inquiry |
        type-param-name | ( expr )
R1002 level-1-expr -> [defined-unary-op] primary
R1003 defined-unary-op -> . letter [letter]... .
R1004 mult-operand -> level-1-expr [power-op mult-operand]
R1005 add-operand -> [add-operand mult-op] mult-operand
R1006 level-2-expr -> [[level-2-expr] add-op] add-operand
R1007 power-op -> **
R1008 mult-op -> * | /
R1009 add-op -> + | -
R1010 level-3-expr -> [level-3-expr concat-op] level-2-expr
R1011 concat-op -> //
R1012 level-4-expr -> [level-3-expr rel-op] level-3-expr
R1013 rel-op ->
        .EQ. | .NE. | .LT. | .LE. | .GT. | .GE. |
        == | /= | < | <= | > | >=  @ | <>
R1014 and-operand -> [not-op] level-4-expr
R1015 or-operand -> [or-operand and-op] and-operand
R1016 equiv-operand -> [equiv-operand or-op] or-operand
R1017 level-5-expr -> [level-5-expr equiv-op] equiv-operand
R1018 not-op -> .NOT.
R1019 and-op -> .AND.
R1020 or-op -> .OR.
R1021 equiv-op -> .EQV. | .NEQV.
R1022 expr -> [expr defined-binary-op] level-5-expr
R1023 defined-binary-op -> . letter [letter]... .
R1024 logical-expr -> expr
R1025 default-char-expr -> expr
R1026 int-expr -> expr
R1027 numeric-expr -> expr
R1028 specification-expr -> scalar-int-expr
R1029 constant-expr -> expr
R1030 default-char-constant-expr -> default-char-expr
R1031 int-constant-expr -> int-expr
R1032 assignment-stmt -> variable = expr
R1033 pointer-assignment-stmt ->
        data-pointer-object [( bounds-spec-list )] => data-target |
        data-pointer-object ( bounds-remapping-list ) => data-target |
        proc-pointer-object => proc-target
R1034 data-pointer-object ->
        variable-name | scalar-variable % data-pointer-component-name
R1035 bounds-spec -> lower-bound-expr :
R1036 bounds-remapping -> lower-bound-expr : upper-bound-expr
R1037 data-target -> expr
R1038 proc-pointer-object -> proc-pointer-name | proc-component-ref
R1039 proc-component-ref -> scalar-variable % procedure-component-name
R1040 proc-target -> expr | procedure-name | proc-component-ref
R1041 where-stmt -> WHERE ( mask-expr ) where-assignment-stmt
R1042 where-construct ->
        where-construct-stmt [where-body-construct]...
        [masked-elsewhere-stmt [where-body-construct]...]...
        [elsewhere-stmt [where-body-construct]...] end-where-stmt
R1043 where-construct-stmt -> [where-construct-name :] WHERE ( mask-expr )
R1044 where-body-construct ->
        where-assignment-stmt | where-stmt | where-construct
R1045 where-assignment-stmt -> assignment-stmt
R1046 mask-expr -> logical-expr
R1047 masked-elsewhere-stmt -> ELSEWHERE ( mask-expr ) [where-construct-name]
R1048 elsewhere-stmt -> ELSEWHERE [where-construct-name]
R1049 end-where-stmt -> END WHERE [where-construct-name]
R1050 forall-construct ->
        forall-construct-stmt [forall-body-construct]... end-forall-stmt
R1051 forall-construct-stmt ->
        [forall-construct-name :] FORALL concurrent-header
R1052 forall-body-construct ->
        forall-assignment-stmt | where-stmt | where-construct |
        forall-construct | forall-stmt
R1053 forall-assignment-stmt -> assignment-stmt | pointer-assignment-stmt
R1054 end-forall-stmt -> END FORALL [forall-construct-name]
R1055 forall-stmt -> FORALL concurrent-header forall-assignment-stmt

R1101 block -> [execution-part-construct]...
R1102 associate-construct -> associate-stmt block end-associate-stmt
R1103 associate-stmt ->
        [associate-construct-name :] ASSOCIATE ( association-list )
R1104 association -> associate-name => selector
R1105 selector -> expr | variable
R1106 end-associate-stmt -> END ASSOCIATE [associate-construct-name]
R1107 block-construct ->
        block-stmt [block-specification-part] block end-block-stmt
R1108 block-stmt -> [block-construct-name :] BLOCK
R1109 block-specification-part ->
        [use-stmt]... [import-stmt]...
        [[declaration-construct]... specification-construct]
R1110 end-block-stmt -> END BLOCK [block-construct-name]
R1111 change-team-construct -> change-team-stmt block end-change-team-stmt
R1112 change-team-stmt ->
        [team-construct-name :] CHANGE TEAM ( team-value
        [, coarray-association-list] [, sync-stat-list] )
R1113 coarray-association -> codimension-decl => selector
R1114 end-change-team-stmt ->
        END TEAM [( [sync-stat-list] )] [team-construct-name]
R1115 team-value -> scalar-expr
R1116 critical-construct -> critical-stmt block end-critical-stmt
R1117 critical-stmt ->
        [critical-construct-name :] CRITICAL [( [sync-stat-list] )]
R1118 end-critical-stmt -> END CRITICAL [critical-construct-name]
R1119 do-construct -> do-stmt block end-do
R1120 do-stmt -> nonlabel-do-stmt | label-do-stmt
R1121 label-do-stmt -> [do-construct-name :] DO label [loop-control]
R1122 nonlabel-do-stmt -> [do-construct-name :] DO [loop-control]
R1123 loop-control ->
        [,] do-variable = scalar-int-expr , scalar-int-expr
          [, scalar-int-expr] |
        [,] WHILE ( scalar-logical-expr ) |
        [,] CONCURRENT concurrent-header concurrent-locality
R1124 do-variable -> scalar-int-variable-name
R1125 concurrent-header ->
        ( [integer-type-spec ::] concurrent-control-list [, scalar-mask-expr] )
R1126 concurrent-control ->
        index-name = concurrent-limit : concurrent-limit [: concurrent-step]
R1127 concurrent-limit -> scalar-int-expr
R1128 concurrent-step -> scalar-int-expr
R1129 concurrent-locality -> [locality-spec]...
R1130 locality-spec ->
        LOCAL ( variable-name-list ) | LOCAL_INIT ( variable-name-list ) |
        SHARED ( variable-name-list ) | DEFAULT ( NONE )
R1131 end-do -> end-do-stmt | continue-stmt
R1132 end-do-stmt -> END DO [do-construct-name]
R1133 cycle-stmt -> CYCLE [do-construct-name]
R1134 if-construct ->
        if-then-stmt block [else-if-stmt block]... [else-stmt block]
        end-if-stmt
R1135 if-then-stmt -> [if-construct-name :] IF ( scalar-logical-expr ) THEN
R1136 else-if-stmt -> ELSE IF ( scalar-logical-expr ) THEN [if-construct-name]
R1137 else-stmt -> ELSE [if-construct-name]
R1138 end-if-stmt -> END IF [if-construct-name]
R1139 if-stmt -> IF ( scalar-logical-expr ) action-stmt
R1140 case-construct -> select-case-stmt [case-stmt block]... end-select-stmt
R1141 select-case-stmt -> [case-construct-name :] SELECT CASE ( case-expr )
R1142 case-stmt -> CASE case-selector [case-construct-name]
R1143 end-select-stmt -> END SELECT [case-construct-name]
R1144 case-expr -> scalar-expr
R1145 case-selector -> ( case-value-range-list ) | DEFAULT
R1146 case-value-range ->
        case-value | case-value : | : case-value | case-value : case-value
R1147 case-value -> scalar-constant-expr
R1148 select-rank-construct ->
        select-rank-stmt [select-rank-case-stmt block]... end-select-rank-stmt
R1149 select-rank-stmt ->
        [select-construct-name :] SELECT RANK ( [associate-name =>] selector )
R1150 select-rank-case-stmt ->
        RANK ( scalar-int-constant-expr ) [select-construct-name] |
        RANK ( * ) [select-construct-name] |
        RANK DEFAULT [select-construct-name]
R1151 end-select-rank-stmt -> END SELECT [select-construct-name]
R1152 select-type-construct ->
        select-type-stmt [type-guard-stmt block]... end-select-type-stmt
R1153 select-type-stmt ->
        [select-construct-name :] SELECT TYPE ( [associate-name =>] selector )
R1154 type-guard-stmt ->
        TYPE IS ( type-spec ) [select-construct-name] |
        CLASS IS ( derived-type-spec ) [select-construct-name] |
        CLASS DEFAULT [select-construct-name]
R1155 end-select-type-stmt -> END SELECT [select-construct-name]
R1156 exit-stmt -> EXIT [construct-name]
R1157 goto-stmt -> GO TO label
R1158 computed-goto-stmt -> GO TO ( label-list ) [,] scalar-int-expr
R1159 continue-stmt -> CONTINUE
R1160 stop-stmt -> STOP [stop-code] [, QUIET = scalar-logical-expr]
R1161 error-stop-stmt -> ERROR STOP [stop-code] [, QUIET = scalar-logical-expr]
R1162 stop-code -> scalar-default-char-expr | scalar-int-expr
R1163 fail-image-stmt -> FAIL IMAGE
R1164 sync-all-stmt -> SYNC ALL [( [sync-stat-list] )]
R1165 sync-stat -> STAT = stat-variable | ERRMSG = errmsg-variable
R1166 sync-images-stmt -> SYNC IMAGES ( image-set [, sync-stat-list] )
R1167 image-set -> int-expr | *
R1168 sync-memory-stmt -> SYNC MEMORY [( [sync-stat-list] )]
R1169 sync-team-stmt -> SYNC TEAM ( team-value [, sync-stat-list] )
R1170 event-post-stmt -> EVENT POST ( event-variable [, sync-stat-list] )
R1171 event-variable -> scalar-variable
R1172 event-wait-stmt -> EVENT WAIT ( event-variable [, event-wait-spec-list] )
R1173 event-wait-spec -> until-spec | sync-stat
R1174 until-spec -> UNTIL_COUNT = scalar-int-expr
R1175 form-team-stmt ->
        FORM TEAM ( team-number , team-variable [, form-team-spec-list] )
R1176 team-number -> scalar-int-expr
R1177 team-variable -> scalar-variable
R1178 form-team-spec -> NEW_INDEX = scalar-int-expr | sync-stat
R1179 lock-stmt -> LOCK ( lock-variable [, lock-stat-list] )
R1180 lock-stat -> ACQUIRED_LOCK = scalar-logical-variable | sync-stat
R1181 unlock-stmt -> UNLOCK ( lock-variable [, sync-stat-list] )
R1182 lock-variable -> scalar-variable

R1201 io-unit -> file-unit-number | * | internal-file-variable
R1202 file-unit-number -> scalar-int-expr
R1203 internal-file-variable -> char-variable
R1204 open-stmt -> OPEN ( connect-spec-list )
R1205 connect-spec ->
        [UNIT =] file-unit-number | ACCESS = scalar-default-char-expr |
        ACTION = scalar-default-char-expr |
        ASYNCHRONOUS = scalar-default-char-expr |
        BLANK = scalar-default-char-expr |
        DECIMAL = scalar-default-char-expr | DELIM = scalar-default-char-expr |
        ENCODING = scalar-default-char-expr | ERR = label |
        FILE = file-name-expr | FORM = scalar-default-char-expr |
        IOMSG = iomsg-variable | IOSTAT = scalar-int-variable |
        NEWUNIT = scalar-int-variable | PAD = scalar-default-char-expr |
        POSITION = scalar-default-char-expr | RECL = scalar-int-expr |
        ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
        STATUS = scalar-default-char-expr
        @ | CARRIAGECONTROL = scalar-default-char-expr
          | CONVERT = scalar-default-char-expr
          | DISPOSE = scalar-default-char-expr
R1206 file-name-expr -> scalar-default-char-expr
R1207 iomsg-variable -> scalar-default-char-variable
R1208 close-stmt -> CLOSE ( close-spec-list )
R1209 close-spec ->
        [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
        IOMSG = iomsg-variable | ERR = label |
        STATUS = scalar-default-char-expr
R1210 read-stmt ->
        READ ( io-control-spec-list ) [input-item-list] |
        READ format [, input-item-list]
R1211 write-stmt -> WRITE ( io-control-spec-list ) [output-item-list]
R1212 print-stmt -> PRINT format [, output-item-list]
R1213 io-control-spec ->
        [UNIT =] io-unit | [FMT =] format | [NML =] namelist-group-name |
        ADVANCE = scalar-default-char-expr |
        ASYNCHRONOUS = scalar-default-char-constant-expr |
        BLANK = scalar-default-char-expr | DECIMAL = scalar-default-char-expr |
        DELIM = scalar-default-char-expr | END = label | EOR = label |
        ERR = label | ID = id-variable | IOMSG = iomsg-variable |
        IOSTAT = scalar-int-variable | PAD = scalar-default-char-expr |
        POS = scalar-int-expr | REC = scalar-int-expr |
        ROUND = scalar-default-char-expr | SIGN = scalar-default-char-expr |
        SIZE = scalar-int-variable
R1214 id-variable -> scalar-int-variable
R1215 format -> default-char-expr | label | *
R1216 input-item -> variable | io-implied-do
R1217 output-item -> expr | io-implied-do
R1218 io-implied-do -> ( io-implied-do-object-list , io-implied-do-control )
R1219 io-implied-do-object -> input-item | output-item
R1220 io-implied-do-control ->
        do-variable = scalar-int-expr , scalar-int-expr [, scalar-int-expr]
R1221 dtv-type-spec -> TYPE ( derived-type-spec ) | CLASS ( derived-type-spec )
R1222 wait-stmt -> WAIT ( wait-spec-list )
R1223 wait-spec ->
        [UNIT =] file-unit-number | END = label | EOR = label | ERR = label |
        ID = scalar-int-expr | IOMSG = iomsg-variable |
        IOSTAT = scalar-int-variable
R1224 backspace-stmt ->
        BACKSPACE file-unit-number | BACKSPACE ( position-spec-list )
R1225 endfile-stmt -> ENDFILE file-unit-number | ENDFILE ( position-spec-list )
R1226 rewind-stmt -> REWIND file-unit-number | REWIND ( position-spec-list )
R1227 position-spec ->
        [UNIT =] file-unit-number | IOMSG = iomsg-variable |
        IOSTAT = scalar-int-variable | ERR = label
R1228 flush-stmt -> FLUSH file-unit-number | FLUSH ( flush-spec-list )
R1229 flush-spec ->
        [UNIT =] file-unit-number | IOSTAT = scalar-int-variable |
        IOMSG = iomsg-variable | ERR = label
R1230 inquire-stmt ->
        INQUIRE ( inquire-spec-list ) |
        INQUIRE ( IOLENGTH = scalar-int-variable ) output-item-list
R1231 inquire-spec ->
        [UNIT =] file-unit-number | FILE = file-name-expr |
        ACCESS = scalar-default-char-variable |
        ACTION = scalar-default-char-variable |
        ASYNCHRONOUS = scalar-default-char-variable |
        BLANK = scalar-default-char-variable |
        DECIMAL = scalar-default-char-variable |
        DELIM = scalar-default-char-variable |
        ENCODING = scalar-default-char-variable |
        ERR = label | EXIST = scalar-logical-variable |
        FORM = scalar-default-char-variable |
        FORMATTED = scalar-default-char-variable | ID = scalar-int-expr |
        IOMSG = iomsg-variable | IOSTAT = scalar-int-variable |
        NAME = scalar-default-char-variable |
        NAMED = scalar-logical-variable | NEXTREC = scalar-int-variable |
        NUMBER = scalar-int-variable | OPENED = scalar-logical-variable |
        PAD = scalar-default-char-variable |
        PENDING = scalar-logical-variable | POS = scalar-int-variable |
        POSITION = scalar-default-char-variable |
        READ = scalar-default-char-variable |
        READWRITE = scalar-default-char-variable |
        RECL = scalar-int-variable | ROUND = scalar-default-char-variable |
        SEQUENTIAL = scalar-default-char-variable |
        SIGN = scalar-default-char-variable | SIZE = scalar-int-variable |
        STREAM = scalar-default-char-variable |
        STATUS = scalar-default-char-variable |
        WRITE = scalar-default-char-variable
        @ | CARRIAGECONTROL = scalar-default-char-expr
          | CONVERT = scalar-default-char-expr
          | DISPOSE = scalar-default-char-expr

R1301 format-stmt -> FORMAT format-specification
R1302 format-specification ->
        ( [format-items] ) | ( [format-items ,] unlimited-format-item )
R1303 format-items -> format-item [[,] format-item]...
R1304 format-item ->
        [r] data-edit-desc | control-edit-desc | char-string-edit-desc | [r] ( format-items )
R1305 unlimited-format-item -> * ( format-items )
R1306 r -> digit-string
R1307 data-edit-desc ->
        I w [. m] | B w [. m] | O w [. m] | Z w [. m] | F w . d |
        E w . d [E e] | EN w . d [E e] | ES w . d [E e] | EX w . d [E e] |
        G w [. d [E e]] | L w | A [w] | D w . d |
        DT [char-literal-constant] [( v-list )]
R1308 w -> digit-string
R1309 m -> digit-string
R1310 d -> digit-string
R1311 e -> digit-string
R1312 v -> [sign] digit-string
R1313 control-edit-desc ->
        position-edit-desc | [r] / | : | sign-edit-desc | k P |
        blank-interp-edit-desc | round-edit-desc | decimal-edit-desc |
        @ $ | \
R1314 k -> [sign] digit-string
R1315 position-edit-desc -> T n | TL n | TR n | n X
R1316 n -> digit-string
R1317 sign-edit-desc -> SS | SP | S
R1318 blank-interp-edit-desc -> BN | BZ
R1319 round-edit-desc -> RU | RD | RZ | RN | RC | RP
R1320 decimal-edit-desc -> DC | DP
R1321 char-string-edit-desc -> char-literal-constant

R1401 main-program ->
        [program-stmt] [specification-part] [execution-part]
        [internal-subprogram-part] end-program-stmt
R1402 program-stmt -> PROGRAM program-name
R1403 end-program-stmt -> END [PROGRAM [program-name]]
R1404 module ->
        module-stmt [specification-part] [module-subprogram-part]
        end-module-stmt
R1405 module-stmt -> MODULE module-name
R1406 end-module-stmt -> END [MODULE [module-name]]
R1407 module-subprogram-part -> contains-stmt [module-subprogram]...
R1408 module-subprogram ->
        function-subprogram | subroutine-subprogram |
        separate-module-subprogram
R1409 use-stmt ->
        USE [[, module-nature] ::] module-name [, rename-list] |
        USE [[, module-nature] ::] module-name , ONLY : [only-list]
R1410 module-nature -> INTRINSIC | NON_INTRINSIC
R1411 rename ->
        local-name => use-name |
        OPERATOR ( local-defined-operator ) =>
          OPERATOR ( use-defined-operator )
R1412 only -> generic-spec | only-use-name | rename
R1413 only-use-name -> use-name
R1414 local-defined-operator -> defined-unary-op | defined-binary-op
R1415 use-defined-operator -> defined-unary-op | defined-binary-op
R1416 submodule ->
        submodule-stmt [specification-part] [module-subprogram-part]
        end-submodule-stmt
R1417 submodule-stmt -> SUBMODULE ( parent-identifier ) submodule-name
R1418 parent-identifier -> ancestor-module-name [: parent-submodule-name]
R1419 end-submodule-stmt -> END [SUBMODULE [submodule-name]]
R1420 block-data -> block-data-stmt [specification-part] end-block-data-stmt
R1421 block-data-stmt -> BLOCK DATA [block-data-name]
R1422 end-block-data-stmt -> END [BLOCK DATA [block-data-name]]

R1501 interface-block ->
        interface-stmt [interface-specification]... end-interface-stmt
R1502 interface-specification -> interface-body | procedure-stmt
R1503 interface-stmt -> INTERFACE [generic-spec] | ABSTRACT INTERFACE
R1504 end-interface-stmt -> END INTERFACE [generic-spec]
R1505 interface-body ->
        function-stmt [specification-part] end-function-stmt |
        subroutine-stmt [specification-part] end-subroutine-stmt
R1506 procedure-stmt -> [MODULE] PROCEDURE [::] specific-procedure-list
R1507 specific-procedure -> procedure-name
R1508 generic-spec ->
        generic-name | OPERATOR ( defined-operator ) |
        ASSIGNMENT ( = ) | defined-io-generic-spec
R1509 defined-io-generic-spec ->
        READ ( FORMATTED ) | READ ( UNFORMATTED ) |
        WRITE ( FORMATTED ) | WRITE ( UNFORMATTED )
R1510 generic-stmt ->
        GENERIC [, access-spec] :: generic-spec => specific-procedure-list
R1511 external-stmt -> EXTERNAL [::] external-name-list
R1512 procedure-declaration-stmt ->
        PROCEDURE ( [proc-interface] ) [[, proc-attr-spec]... ::]
        proc-decl-list
R1513 proc-interface -> interface-name | declaration-type-spec
R1514 proc-attr-spec ->
        access-spec | proc-language-binding-spec | INTENT ( intent-spec ) |
        OPTIONAL | POINTER | PROTECTED | SAVE
R1515 proc-decl -> procedure-entity-name [=> proc-pointer-init]
R1516 interface-name -> name
R1517 proc-pointer-init -> null-init | initial-proc-target
R1518 initial-proc-target -> procedure-name
R1519 intrinsic-stmt -> INTRINSIC [::] intrinsic-procedure-name-list
R1520 function-reference -> procedure-designator ( [actual-arg-spec-list] )
R1521 call-stmt -> CALL procedure-designator [( [actual-arg-spec-list] )]
R1522 procedure-designator ->
        procedure-name | proc-component-ref | data-ref % binding-name
R1523 actual-arg-spec -> [keyword =] actual-arg
R1524 actual-arg ->
        expr | variable | procedure-name | proc-component-ref | alt-return-spec
R1525 alt-return-spec -> * label
R1526 prefix -> prefix-spec [prefix-spec]...
R1527 prefix-spec ->
        declaration-type-spec | ELEMENTAL | IMPURE | MODULE | NON_RECURSIVE |
        PURE | RECURSIVE
R1528 proc-language-binding-spec -> language-binding-spec
R1529 function-subprogram ->
        function-stmt [specification-part] [execution-part]
        [internal-subprogram-part] end-function-stmt
R1530 function-stmt ->
        [prefix] FUNCTION function-name ( [dummy-arg-name-list] ) [suffix]
R1531 dummy-arg-name -> name
R1532 suffix ->
        proc-language-binding-spec [RESULT ( result-name )] |
        RESULT ( result-name ) [proc-language-binding-spec]
R1533 end-function-stmt -> END [FUNCTION [function-name]]
R1534 subroutine-subprogram ->
        subroutine-stmt [specification-part] [execution-part]
        [internal-subprogram-part] end-subroutine-stmt
R1535 subroutine-stmt ->
        [prefix] SUBROUTINE subroutine-name
        [( [dummy-arg-list] ) [proc-language-binding-spec]]
R1536 dummy-arg -> dummy-arg-name | *
R1537 end-subroutine-stmt -> END [SUBROUTINE [subroutine-name]]
R1538 separate-module-subprogram ->
        mp-subprogram-stmt [specification-part] [execution-part]
        [internal-subprogram-part] end-mp-subprogram-stmt
R1539 mp-subprogram-stmt -> MODULE PROCEDURE procedure-name
R1540 end-mp-subprogram-stmt -> END [PROCEDURE [procedure-name]]
R1541 entry-stmt -> ENTRY entry-name [( [dummy-arg-list] ) [suffix]]
R1542 return-stmt -> RETURN [scalar-int-expr]
R1543 contains-stmt -> CONTAINS
R1544 stmt-function-stmt ->
        function-name ( [dummy-arg-name-list] ) = scalar-expr
```        
