#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ps13: ps13.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
ps13.$(OBJX): $(SRC)/ps13.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ps13.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) ps13.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ps13
run: ;
