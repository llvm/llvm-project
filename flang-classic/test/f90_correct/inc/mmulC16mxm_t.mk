#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulC16mxm_t  ########


mmulC16mxm_t: run
	

build:  $(SRC)/mmulC16mxm_t.f90
	-$(RM) mmulC16mxm_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulC16mxm_t.f90 -o mmulC16mxm_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulC16mxm_t.$(OBJX) check.$(OBJX) $(LIBS) -o mmulC16mxm_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulC16mxm_t
	mmulC16mxm_t.$(EXESUFFIX)

verify: ;

mmulC16mxm_t.run: run

