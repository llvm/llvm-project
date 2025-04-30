#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR8mxm_t  ########


mmulR8mxm_t: run
	

build:  $(SRC)/mmulR8mxm_t.f90
	-$(RM) mmulR8mxm_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR8mxm_t.f90 -o mmulR8mxm_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR8mxm_t.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR8mxm_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR8mxm_t
	mmulR8mxm_t.$(EXESUFFIX)

verify: ;

mmulR8mxm_t.run: run

