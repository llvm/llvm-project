#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test em00  ########


em00: run
	

build:  $(SRC)/em00.f
	-$(RM) em00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/em00.f -o em00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) em00.$(OBJX) check.$(OBJX) $(LIBS) -o em00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test em00
	em00.$(EXESUFFIX)

verify: ;

em00.run: run

