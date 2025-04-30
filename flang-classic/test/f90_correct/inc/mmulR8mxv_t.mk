#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test mmulR8mxv_t  ########


mmulR8mxv_t: run
	

build:  $(SRC)/mmulR8mxv_t.f90
	-$(RM) mmulR8mxv_t.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/mmulR8mxv_t.f90 -o mmulR8mxv_t.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) mmulR8mxv_t.$(OBJX) check.$(OBJX) $(LIBS) -o mmulR8mxv_t.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test mmulR8mxv_t
	mmulR8mxv_t.$(EXESUFFIX)

verify: ;

mmulR8mxv_t.run: run

