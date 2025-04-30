#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test nb10  ########


nb10: run
	

build:  $(SRC)/nb10.f
	-$(RM) nb10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/nb10.f -o nb10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) nb10.$(OBJX) check.$(OBJX) $(LIBS) -o nb10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test nb10
	nb10.$(EXESUFFIX)

verify: ;

nb10.run: run

