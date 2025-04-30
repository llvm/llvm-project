#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test cb10  ########


cb10: run
	

build:  $(SRC)/cb10.f
	-$(RM) cb10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/cb10.f -o cb10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) cb10.$(OBJX) check.$(OBJX) $(LIBS) -o cb10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test cb10
	cb10.$(EXESUFFIX)

verify: ;

cb10.run: run

