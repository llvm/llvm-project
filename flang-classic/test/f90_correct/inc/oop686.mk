# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

oop686: run
build: $(SRC)/oop686.f90
	-$(RM) oop686.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test oop686
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop686.f90 -o oop686.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop686.$(OBJX) check.$(OBJX) $(LIBS) -o oop686.$(EXESUFFIX)
run:
	@echo ------------------------------------ executing test oop686
	./oop686.$(EXESUFFIX)
verify: ;

