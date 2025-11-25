//! Thinking mode and extended reasoning

pub struct ThinkingMode {
    pub enabled: bool,
}

impl ThinkingMode {
    pub fn new() -> Self {
        Self { enabled: false }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn toggle(&mut self) {
        self.enabled = !self.enabled;
    }
}

impl Default for ThinkingMode {
    fn default() -> Self {
        Self::new()
    }
}
