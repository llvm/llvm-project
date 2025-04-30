
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

	VRS_T rarg1;
	VRS_T rout;
	int   nfails = 0;
	int   nfailsm = 0;
	char  *fname;

#if MAX_VREG_SIZE != 64
	VRS_T routm;
	int   imask, im;
	char  *fnamem;
#endif

	fname = STRINGIFY(CONCAT7(__,FRP,PREC,_,FUNC,_,VLS));

	rarg1 = vrs_set_arg(FMIN, FCONST1);
	rout = CONCAT7(__,FRP,PREC,_,FUNC,_,VLS)(rarg1);
        nfails += checkfltol(rout,expd_res,vmask,VLS,TOL);

#if MAX_VREG_SIZE != 64
	fnamem = STRINGIFY(CONCAT8(__,FRP,PREC,_,FUNC,_,VLS,m));

	for (imask = 0 ; imask < 2 ; imask++) {
            build_masks(imask==1);
	    for(im = 0 ; im < 1<<VLS; im++) {
	        vmask = *(VIS_T *)&mask_sp[im];
	        routm = CONCAT8(__,FRP,PREC,_,FUNC,_,VLS,m)(rarg1,vmask);
		nfailsm += checkfltol(routm,expd_res,vmask,VLS,TOL);
	    }
	}
#endif

	if (nfails != 0) {
	    printf("%s Test Failed\n",fname);
	} else {
	    printf("%s Test Passed\n",fname);
	}

#if MAX_VREG_SIZE != 64
	if (nfailsm != 0) {
	    printf("%s Test Failed\n",fnamem);
	} else {
	    printf("%s Test Passed\n",fnamem);
	}
#endif

	if ((nfails != 0) || (nfailsm != 0)) {
	    exit(-1);
	}
