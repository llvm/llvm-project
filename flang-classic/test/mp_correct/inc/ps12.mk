#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
ps12: ps12.$(OBJX)
	@echo ------------ executing test $@
	-$(RUN2) ./a.$(EXESUFFIX) $(LOG)
ps12.$(OBJX): $(SRC)/ps12.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/ps12.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) ps12.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: ps12
run: ;
