#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
mul01: mul01.$(OBJX)
	@echo ------------ executing test $@
	$(RUN2) a.$(EXESUFFIX) $(LOG)
mul01.$(OBJX): $(SRC)/mul01.f check.$(OBJX)
	$(FC) $(FFLAGS) $(SRC)/mul01.f
	@$(RM) a.$(EXESUFFIX)
	$(FC) $(LDFLAGS) mul01.$(OBJX) check.$(OBJX) $(LIBS) -o a.$(EXESUFFIX)
build: mul01
run: ;
