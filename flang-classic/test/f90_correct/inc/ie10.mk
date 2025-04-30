#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ie10  ########


ie10: run
	

build:  $(SRC)/ie10.f
	-$(RM) ie10.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ie10.f -o ie10.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ie10.$(OBJX) check.$(OBJX) $(LIBS) -o ie10.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ie10
	ie10.$(EXESUFFIX)

verify: ;

ie10.run: run

