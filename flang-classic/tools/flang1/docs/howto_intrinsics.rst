

********************************
How to add an intrinsic to flang
********************************

This page contains a guide of how to add a new function/intrinsic to FLANG library. It was based on the intrinsics ``TRAILZ`` and ``LEADZ``, which are math runtime intrinsics. Therefore all the steps may need some changes based on which type of intrinsic is being added to FLANG library. The function/intrinsic will be added inside ``<directory_where_cloned>/flang/runtime/flang``. ``runtime`` is the fortran runtime library.

Step-by-step guide
##################

How to link a new intrinsic in flang?

#. Create the .c (``your_intrinsic.c``) file in the directory e.g. ``runtime/flang``. It is common sense to create one file for each data type. For instance, numbers can be expressed as ``integers``, ``long int``, ``short int`` and so on. So in this case it is better to create one file for each intrinsic. Each one of these intrinsics has a different input according to its data type.

#. In the directory ``tools/flang1/utils/symtab`` there are 2 files ``symini_ftn.n`` and ``symini.cpp`` that need to be changed. Symini adds a reference of the new intrinsic in the AST (Abstract Syntaxe Tree) using ``symini.cpp`` and ``symini_ft.n``.

    a. In ``tools/flang1/utils/symtab/symini_ftn.n`` add the lines below where it is suitable. This file is needed by ``ast.n`` because it has the symbols of the intrinsics to build the ILM file. You can look into the file ``symini_ftn.n`` to understand the syntax of the file.

    .. code-block:: c

        i. .AT elemental I
        .H1 YOUR_INTRINSIC –
        ii. .AT elemental i
        .H7 YOUR_INTRINSIC --

    b. In file ``tools/flang1/utils/symtab/symini.cpp`` find array ``const char *SyminiFE90::init_namesX[]`` (where ``x`` is a number) add your intrinsic between brackets ``" your_intrinsic "`` . Do not know which one of the ``X``s (``SyminiFE90::init_namesX[]``) is the most correct, but I believe any one of them will work.
	
#. ``tools/flang1/flang1exe/``. Basically there are 3 files to that you need to modify, they are: ``lowerexp.c``, ``stout.c`` and ``semfunc.c`` . The changes will allow flang1 to build the ILM file. ILM is the intermediate file between Fortran and flang2.

    a. ``/tools/flang1/flang1exe/lowerexp.c`` has routines used by ``lower.c`` for lowering to ILMs. ``lower.c`` creates the back-end ILM structure. It lowers the AST and the ILM for something that the next layer can use. You must add the reference of your intrinsic in two functions:

        i. In ``static int intrinsic_arg_dtype(int intr, int ast, int args, int nargs)`` add the line bellow and change ``YOUR_INTRINSIC`` for the name of the function you have implemented in upper case.
	
        .. code-block:: c

            case I_ YOUR_INTRINSIC :

        ii. In ``static int lower_intrinsic(int ast)`` , add the line below and change ``YOUR_INTRINSIC`` for the name of the intrinsic you have implemented in upper case. The function ``intrin_name_bsik (char *name, int ast)`` recognises the type of the data that follows the intrinsic and writes in the ILM.

        .. code-block:: c

            case I_ YOUR_INTRINSIC :
                ilm = intrin_name_bsik(" YOUR_INTRINSIC ", ast);
                break;

    b. In file ``/tools/flang1/flang1exe/astout.c`` in the function ``static void print_as t(int ast)``:

    .. code-block:: c
        
        #ifdef I_YOUR_INTRINSIC
        case I_ YOUR_INTRINSIC :
        if (XBIT(49, 0x1040000)) 
        /* T3D/T3E or C90 Cray targets */
        put_call(ast, 0, NULL, 0);
        break;
        }
        
        rtlRtn = RTE_ your_intrinsic ;
        goto make_func_name;
        #endif

    c. ``/tools/flang1/flang1exe/semfunc.c`` is Fortran front-end utility routines used by Semantic Analyzer to process functions, subroutines, predeclares, etc. In the function ``int ref_pd(SST *stktop, ITEM *list)`` add the line below. The goal is to write in the ILM the pre-defined function.
        
    .. code-block:: c

        case PD_ your_intrinsic :

#. The directory ``tools/shared`` has functions that provide the front-end access to the runtime library structure. You need also to change the files ``rtlRtns.h`` and ``rtlRtns.c``. ``rtlRtns.h`` has the Enumerator for some (eventually all) the RTL library routines. ``rtlRtns.c`` has the entries that must be sorted on the ``baseNm`` field. *NOTE:* make sure they are in the same order in the list, otherwise it may generate an error.

    a. ``/tools/shared/rtlRtns.h`` in ``typedef enum {`` add:

    .. code-block:: c

        RTE_your_intrinsic
    
    b. ``/tools/shared/rtlRtns.c`` in ``FtnRteRtn ftnRtlRtns[] = {`` add:
   
    .. code-block:: c
         
        {" your_intrinsic ", "", FALSE, ""},

#. ``tools/flang2/flang2exe`` is responsible for the front-end of the stage 2[1], therefore ``flang2exe`` files are the ones that change the file from ILM to ILI and at the end to LLVM IR. In this directory you may need to change some files as well. Have a look at ``math.h``, ``iliutil.cpp`` and ``exp_ftn.cpp``.

    a. ``/tools/flang2/flang2exe/mth.h`` this library parameterizes the names of the ``__mth_i/__fmth_i ...`` functions. In case the intrinsic has different behaviour for different inputs, then you should create one ``#define`` function for each behavior, for instance:
    
    .. code-block:: c 
    
        #define MTH_I_I YOUR_INTRINSIC I "__mth_i_i your_intrinsic i". //for data type of 8 or 16 bits
        #define MTH_I_I YOUR_INTRINSIC "__mth_i_i your_intrinsic ". //for data type equals to 32 bits
        #define MTH_I_K YOUR_INTRINSIC "__mth_i_k your_intrinsic " //for data type equals to 64 bits

    b. ``/tools/flang2/flang2exe/iliutil.cpp`` has the ILI utility module. They contain the reference of your intrinsics created at ``mth.h``. ``void prilitree(int i) {`` add reference to the intrinsic. The ``goto intrinsic`` writes on the ``gbl.dbgfi``. Add one case statement for each ``mth_i`` created on ``mth.h``
   
    .. code-block:: c 
   
       case IL_I YOUR_INTRINSIC I: //mth_i_i your_intrinsic i
       n = 2;
       opval = " your_intrinsic ";
       goto intrinsic;
       case IL_I YOUR_INTRINSIC : //mth_i_i your_intrinsic
       case IL_K YOUR_INTRINSIC : //mth_i_k your_intrinsic
       n = 1;
       opval = " your_intrinsic ";
       goto intrinsic;

    c. ``/tools/flang2/flang2exe/exp_ftn.cpp`` has the Fortran-specific expander routines. In ``void exp_ac(ILM_OP opc, ILM *ilmp, int curilm)`` add the following lines. ``exp_ac()`` is called in ``eval_ilm()`` in ``expand.cpp`` . It looks like ``expand.cpp`` file works on top of ILM and will expand it.
    
    .. code-block:: c
        
        case IM_K YOUR_INTRINSIC :
        op1 = ILI_OF(ILM_OPND(ilmp, 1));
        //takes the op from ILM to build the ILI
        if (XBIT(124, 0x400)).
        //124 is a mask and 0x400: 64 bits of precision for integer*8 and logical*8 operations [2].
        ilix = ad1ili(IL_K YOUR_INTRINSIC , op1); // add op to ILI ad1ili(ILI_OP opc, int opn1 ) in iliutil.cpp
        else {
        op1 = kimove(op1);
        // kimove(int ilix) in iliutil.cpp
        ilix = ad1ili(IL_I YOUR_INTRINSIC , op1);
        }
        ILM_RESULT(curilm) = ilix.
        
        
#. ``/tools/flang2/utils/ilitp`` has 3 more directories, one for each microprocessor architecture where Flang may be used : aarch64, ppc64le and x86_64 . In each of these directories there are files ``ilitp.n``, ``ilitp_longdouble.n``. According to what you are doing you need to change both. But usually you should change only ``ilitp.n`` in each one of the directories (aarch64, ppc64le and x86_64). They are responsible for writing ILI optimised for each architecture. The ``iliutil.cpp`` reads the ILI file with the help of ``ilitp.n``.
    
    .. code-block:: c
    
        IL I YOUR_INTRINCIS I irlnk stc
        8-/16- bit integer YOUR_INTRINSIC intrinsic.
        The value, 0 or 1, of the second operand indicates
        8-bit or 16-bit, respectively.
        .AT arth null ir cse
        
        .IL I YOUR_INTRINSIC irlnk
        32-bit integer YOUR_INTRINSIC intrinsic.
        .AT arth null ir cse
        .CG "?????" 'l'
        
        .IL K YOUR_INTRINSIC krlnk
        64-bit integer YOUR_INTRINSIC intrinsic.
        .AT arth null kr cse
        .CG "??????t" 'q'
#. ``/tools/flang2/utils/ilmtp/`` has also 3 directories one for each architecture aarch64, xb6_64 and ppc64le . In these you will find ``ilmtp.n``. You need to modify this file as well by adding the following lines:

    .. code-block:: c
        
        .IL B YOUR_INTRINSIC intr lnk
        8-bit integer YOUR_INSTRINSIC intrinsic
        .OP I YOUR_INSTRINSIC I r p1 iv0
        .IL S YOUR_INSTRINSIC intr lnk
        16-bit integer YOUR_INSTRINSIC intrinsic
        .OP I YOUR_INSTRINSIC I r p1 iv1
        .IL I YOUR_INSTRINSIC intr lnk
        32-bit integer YOUR_INTRINSIC intrinsic
        .OP I YOUR_INSTRINSIC r p1
        .IL K YOUR_INSTRINSIC intr lnk
        64-bit integer YOUR_INSTRINSIC intrinsic
        .AT i8
        .OP K YOUR_INSTRINSIC r p1

#. After all the modifications made above, it is needed to check if it works. First run make files in build-flang. Type the commands :

    a. ``make -j48``

    b. ``make install``

#. If you pass the make then it is time to build the test for the new intrinsic/function. I suggest before adding the test on flang, to create a local directory and write a small test for your new intrinsic. The directory ``flang/test/f90_correct`` has the tests for intrinsics. They run with ``make check-all``. Inside ``f90_correct`` there are 3 more directories, they are: ``inc``, ``lit`` and ``src`` . ``src`` has fortran tests in ``.f90``, in ``inc`` is the bash script to call the test function and run it, and ``inc`` has sh files.

#. After you have created your tests build flang again and type:
    
    a. ``make -j48``
    
    b. ``make install``
    
    c. ``make check-all``, this last command will run the tests on ``f90_correct`` and if something went wrong it will tell.

#. When the new intrinsic passed all the tests you can commit for review with the following commands:

    a. ``git add .``
    
    b. ``git commit -m "message of the commit"`` or ``git commit --amend``; the first when it is the first time you commit and the second when you are just updating.

    c. ``git pull --rebase``;

    d. ``git review``
    
#. You should check if the changes done in rebase are not affected. If so change what needs to be changed and apply step 10 again.
