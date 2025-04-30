#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR4mxm_t  ########


mmulR4mxm_t: run
	

build:  $(SRC)/mmulR4mxm_t.f90
	-$(RM) mmulR4mxm_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR4mxm_t.f90 -o mmulR4mxm_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR4mxm_t.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR4mxm_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR4mxm_t
	mmulR4mxm_t.$(EXESUFFIX)

verify: ;

mmulR4mxm_t.run: run

