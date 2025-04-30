#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test gg10  ########


gg10: run
	

build:  $(SRC)/gg10.f
	-$(RM) gg10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/gg10.f -o gg10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) gg10.$(OBJX) check.$(OBJX) $(LIBS) -o gg10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test gg10
	gg10.$(EXESUFFIX)

verify: ;

gg10.run: run

